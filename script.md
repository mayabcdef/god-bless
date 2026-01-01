# Build image (only needed once or when requirements change)
docker build -f docker/Dockerfile -t forge3d-pipeline:latest .

# Production run (code baked into image, no auto-reload)
docker run --gpus all -p 10006:10006 \
  -e UVICORN_RELOAD=false \
  forge3d-pipeline:latest

# Development run (live code mounting + auto-reload on file changes!)
docker run --gpus all -p 10006:10006 \
  -v $(pwd)/pipeline_service:/workspace \
  forge3d-pipeline:latest