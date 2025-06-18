#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t dutch-vocab-bot .

# Run the container with .env file
echo "Starting Dutch Vocabulary Bot..."
docker run -it --rm \
  --name dutch-vocab-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  dutch-vocab-bot