#!/bin/bash
retro-contest run --agent $DOCKER_REGISTRY/gapo:v2 \
    --results-dir results --use-host-data \
    SonicTheHedgehog-Genesis GreenHillZone.Act1
docker build -f rainbow.docker -t $DOCKER_REGISTRY/gapo:v2 .
