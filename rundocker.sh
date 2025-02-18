torchrun --nnodes=${WORLD_SIZE} --nproc-per-node=2 \
                  --node-rank=${RANK} \
                  --master-addr=${MASTER_ADDR} \
                  --master-port=${MASTER_PORT} \
                   /pfs/sshare/app/saiwanming/CogVLM/webdb.py \
                   --from_pretrained /pfs/sshare/app/saiwanming/models/cogvlm-chat \
                   --target_path  /pfs/sshare/app/saiwanming/data/laion-high-aesthetics_6_recaption \
                   --source_path /pfs/sshare/app/dataset/laion-high-aesthetics_6 \
                   --local_tokenizer /pfs/sshare/app/saiwanming/models/vicuna-7b-v1.5 \
                   --version chat --english --bf16 --run_mode torchrun