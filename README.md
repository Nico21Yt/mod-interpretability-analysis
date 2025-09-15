docker run -it --gpus all \
  --shm-size="16g" \
  -v "$(pwd)":/app \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /app \
  --entrypoint /bin/bash \
  edge-circuit-base