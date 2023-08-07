#!/bin/bash

# Project name, used as a container name
PROJECT_NAME="pixelai"

# Path to the current directory
CURRENT_DIR=$(pwd)

# Allow access to the X server
xhost +local:docker

# Check if a container with the same name is already running
if [ $(docker ps -q -f name=$PROJECT_NAME) ]; then
    echo "Container with the name $PROJECT_NAME is already running."
    echo "Entering the existing container..."
    docker exec -it $PROJECT_NAME bash
fi

# Run the Docker container
docker run --name $PROJECT_NAME \
           --gpus all \
           -e DISPLAY=$DISPLAY \
           -e QT_X11_NO_MITSHM=1 \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $CURRENT_DIR:/PixelAI \
           --privileged \
           -it pixelai

# Revoke access to the X server
xhost -local:docker

# Explanation:
# xhost +local:docker: Allows the Docker container to connect to the host's X server
# -e QT_X11_NO_MITSHM=1: Disables MIT-SHM extension, which can cause issues with some X11 applications
