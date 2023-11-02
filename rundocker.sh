torchrun --nnodes=${WORLD_SIZE} --nproc-per-node=8 \
                  --node-rank=${RANK} \
                  --master-addr=${MASTER_ADDR} \
                  --master-port=${MASTER_PORT} \
                   /pfs/sshare/app/saiwanming/CogVLM/webdb.py \
                   --from_pretrained /pfs/sshare/app/saiwanming/models/cogvlm-chat \
                   --target_path  /pfs/sshare/app/saiwanming/data/laion-high-aesthetics_6_recaption
                   --source_path /pfs/sshare/app/dataset/laion-high-aesthetics_6
                   --version chat --english --bf16 --run_mode torchrun