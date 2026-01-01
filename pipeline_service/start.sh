#!/bin/bash
set -e

# --- Cấu hình mặc định ---
VLLM_HOST="0.0.0.0"
VLLM_PORT=${VLLM_PORT:-8095}
VLLM_MODEL=${VLLM_MODEL:-"zai-org/GLM-4.1V-9B-Thinking"}
GPU_UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.4} 
API_KEY=${VLLM_API_KEY:-"local"}

echo "-----------------------------------------------------"
echo "🚀 STARTING VLLM SERVER (Isolated Env)"
echo "   Model: $VLLM_MODEL"
echo "   Port: $VLLM_PORT"
echo "   GPU Util: $GPU_UTIL"
echo "-----------------------------------------------------"

/opt/vllm-env/bin/vllm serve $VLLM_MODEL \
    --port "8095" \
    --api-key "local" \
    --max-model-len 8096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_UTIL \
    --max_num_seqs 2 &


VLLM_PID=$!

echo "⏳ Waiting for vLLM to become ready..."
MAX_RETRIES=150
COUNTER=0

while [ $COUNTER -lt $MAX_RETRIES ]; do
    # Curl kiểm tra health endpoint
    if curl -s -f "http://localhost:$VLLM_PORT/health" > /dev/null; then
        echo "✅ vLLM is READY!"
        break
    fi
    
    echo "   ... loading model ($COUNTER/$MAX_RETRIES)"
    sleep 5
    let COUNTER=COUNTER+1
done

if [ $COUNTER -eq $MAX_RETRIES ]; then
    echo "❌ vLLM failed to start within timeout. Check /var/log/vllm.log"
    kill $VLLM_PID
    exit 1
fi

echo "-----------------------------------------------------"
echo "🚀 STARTING MAIN FASTAPI SERVICE (Base Env)"
echo "-----------------------------------------------------"

uvicorn serve:app --host 0.0.0.0 --port 10006 --reload
