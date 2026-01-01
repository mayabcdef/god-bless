# 3D Generation Pipeline

Automated pipeline for generating 3D models from 2D images.

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with CUDA 12.x support
- At least **80GB VRAM** (61GB+ recommended)

## Installation

### Docker (building)
```bash
docker build -f docker/Dockerfile -t forge3d-pipeline:latest .
```

## Run pipeline

Copy `.env.sample` to `.env` and configure if needed

- Start with docker-compose 

```bash
cd docker
docker-compose up -d --build
```

- Start with docker run
```bash
docker run --gpus all -p 10006:10006 forge3d-pipeline:latest
```

- Start with docker run and env file
```bash
docker run --gpus all -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

- Start with docker run and env file and bound directory (Useful for active development)
```bash
docker run --gpus all -v ./pipeline_service:/workspace/pipeline_service -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

## API Usage

**Seed parameter:**
- `seed: 42` - Use specific seed for reproducible results
- `seed: -1` - Auto-generate random seed (default)

### Endpoint 1: File upload (returns binary PLY)

```bash
curl -X POST "http://localhost:10006/generate" \
  -F "prompt_image_file=@cr7.png" \
  -F "seed=42" \
  -o model.ply
```

### Endpoint 2: File upload (returns binary SPZ)

```bash
curl -X POST "http://localhost:10006/generate-spz" \
  -F "prompt_image_file=@image.png" \
  -F "seed=42" \
  -o model.spz
```

### Endpoint 3: Base64 (returns JSON)

```bash
curl -X POST "http://localhost:10006/generate_from_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_type": "image",
    "prompt_image": "<base64_encoded_image>",
    "seed": 42
  }'
```

### Endpoint 4: Health check (returns JSON)

```bash
curl -X GET "http://localhost:10006/health" \
  -H "Content-Type: application/json" 
```