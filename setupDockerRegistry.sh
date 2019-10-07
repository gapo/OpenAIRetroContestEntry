#!/bin/bash

# set python to relax
unset PYTHONPATH
# Do docker stuff
export DOCKER_REGISTRY="retrocontestaqvwfedlbknmhszu.azurecr.io"
docker login $DOCKER_REGISTRY \
    --username "retrocontestaqvwfedlbknmhszu" \
    --password "ihqGRfBtTGgGqv+jYrz/g2U8SZks80cl"
