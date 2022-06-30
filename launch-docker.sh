#!/bin/bash
# Builds a Docker image with Python and CUDA and launches Bash
set -e -o xtrace

GPUS=16
IMAGE_NAME=witiko/arqmath3-openqa-tools:latest

DOCKER_BUILDKIT=1 docker build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" --build-arg UNAME="$(id -u -n)" . -t "$IMAGE_NAME"
docker run --rm -it -u "$(id -u):$(id -g)" --runtime=nvidia -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_VISIBLE_DEVICES="$GPUS" -v "$PWD"/..:/workdir:rw -w /workdir "$IMAGE_NAME" bash
