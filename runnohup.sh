#!/usr/bin/env bash

set -e
# set -x

self_world_size=${1:-"1"}
self_rank=${2:-"0"}
local_rank=${3:-"0"}
CUDA_VISIBLE_DEVICES=${local_rank} nohup  python webdb.py --from_pretrained /ML-A100/sshare-app/saiwanming/models/cogvlm-chat --self_rank ${self_rank} --self_world_size ${self_world_size} --version chat  --english --bf16 > app${self_rank}.log 2>&1 &