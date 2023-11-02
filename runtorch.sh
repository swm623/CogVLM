#!/usr/bin/env bash

set -e
# set -x

torchrun --standalone --nnodes=1 --nproc-per-node=2 webdb.py --from_pretrained /ML-A100/sshare-app/saiwanming/models/cogvlm-chat --version chat --english --bf16 --run_mode torchrun