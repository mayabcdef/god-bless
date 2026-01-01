import io
import base64
import asyncio
import httpx
import torch
import numpy as np
from PIL import Image
from typing import Literal, Tuple, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
import time
from logger_config import logger
from renderers.gs_renderer.renderer import Renderer
from renderers.ply_loader import PlyLoader

# --- SETTING ---
VLLM_PORT="8095"
VLLM_MODEL="zai-org/GLM-4.1V-9B-Thinking"
API_KEY="local"
TEMPERATURE = 0.0
MAX_TOKENS = 1024

# --- CONSTANTS RENDER ---
IMG_WIDTH = 518
IMG_HEIGHT = 518
GRID_VIEW_GAP = 5
VIEWS_NUMBER = 16
THETA_ANGLES = np.linspace(0, 360, num=VIEWS_NUMBER)
PHI_ANGLES = np.full_like(THETA_ANGLES, -15.0)
GRID_VIEW_INDICES = [1, 5, 9, 13]
CAM_RAD = 2.5
CAM_FOV_DEG = 49.1
REF_BBOX_SIZE = 1.5

# --- CONSTANTS JUDGE ---
SYSTEM_PROMPT = """
You are a specialized 3D model evaluation system. 
Analyze visual quality and prompt adherence with expert precision. 
Always respond with valid JSON only."""
USER_PROMPT_IMAGE = """Does each 3D model match the image prompt?

Penalty 0-10:
0 = Perfect match
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing)
7 = Major issues (wrong category but related, e.g. chair vs stool)
10 = Completely wrong object

Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}"""

class JudgeResponse(BaseModel):
    """Response from a judge evaluating a duel between two models."""

    penalty_1: int
    """Penalty for the first model (0-10, lower is better)."""
    penalty_2: int
    """Penalty for the second model (0-10, lower is better)."""
    issues: str
    """Human-readable issue summary produced by the judge."""

class DuelResponse(BaseModel):
    winner: int
    issues: tuple[str, str]
    score1: tuple[int, int]
    score2: tuple[int, int]
    prompt: bytes
    img1: bytes
    img2: bytes


class DuelManager:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key=API_KEY,
            http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))
        )

    async def _call_vllm(self, prompt_b64: str, img1_b64: str, img2_b64: str) -> JudgeResponse:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image prompt to generate 3D model:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prompt_b64}"}},
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                ],
            },
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }

        try:
            start = time.time()
            completion = await self.client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS, 
                response_format=response_format,
            )
            end = time.time()
            logger.info(f"vLLM call time: {end - start} seconds")
            
            # Log stop reason for debugging
            logger.info(f"vLLM finish_reason: {completion.choices[0].finish_reason}")
            return JudgeResponse.model_validate_json(completion.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"vLLM Call Failed: {e}")
            if 'completion' in locals():
                logger.error(f"Response content: {completion.choices[0].message.content[:500]}")
            return JudgeResponse(penalty_1=-1, penalty_2=-1, issues="JSON Parse Error")
        
    async def run_duel(self, prompt_bytes: bytes, img1_bytes: bytes, img2_bytes: bytes) -> DuelResponse:
        """
        Return DuelRespone with winner = [1,0,-1] mean img1 win, draw or lost img2
        """
        
        # 1. Prepare Base64
        prompt_b64 = base64.b64encode(prompt_bytes).decode('utf-8').strip()
        render1_b64 = base64.b64encode(img1_bytes).decode('utf-8').strip()
        render2_b64 = base64.b64encode(img2_bytes).decode('utf-8').strip()

        # 2. Position-Balanced Duel (Đấu 2 lượt đảo vị trí)
        logger.info("Asking Judge (vLLM)...")
        res_direct, res_swapped = await asyncio.gather(
            self._call_vllm(prompt_b64, render1_b64, render2_b64),
            self._call_vllm(prompt_b64, render2_b64, render1_b64)
        )

        # 3. Handle both dict and JudgeResponse objects
        score1 = (res_direct.penalty_1, res_swapped.penalty_2)
        score2 = (res_direct.penalty_2, res_swapped.penalty_1)
        issues = (res_direct.issues, res_swapped.issues)
        
        s1 = (score1[0] + score1[1]) / 2
        s2 = (score2[0] + score2[1]) / 2
        if abs(s1 - s2) <= 1:
            winner = 0
        elif s1 > s2:
            winner = 1
        else:
            winner = -1
            
        return DuelResponse(winner=winner, issues=issues, score1=score1, score2=score2,
                            prompt=prompt_bytes, img1=img1_bytes, img2=img2_bytes)
        
duel_manager = DuelManager()