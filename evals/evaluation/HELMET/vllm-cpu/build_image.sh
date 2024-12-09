# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


echo "Building the vllm-cpu docker images"
cd $WORKDIR
echo $WORKDIR

if [ ! -d "./vllm" ]; then
    git clone https://github.com/vllm-project/vllm.git
fi
cd ./vllm
git checkout main

DOCKER_BUILDKIT=1 docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
if [ $? -ne 0 ]; then
    echo "vllm-cpu-env failed"
    exit 1
else
    echo "vllm-cpu-env successful"
fi