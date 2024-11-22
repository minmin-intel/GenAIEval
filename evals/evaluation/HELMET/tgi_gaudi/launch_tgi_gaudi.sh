# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-70B-Instruct"
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export HF_CACHE_DIR=${HF_CACHE_DIR}
export MAX_INPUT=65536
export MAX_TOTAL=131072
docker compose -f compose.yaml up -d
