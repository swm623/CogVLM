export CUDA_VISIBLE_DEVICES=7
python cli_demo.py --from_pretrained /home/saiwanming/models/cogvlm-chat --version chat  --english --bf16
python webdb.py --from_pretrained /home/saiwanming/models/cogvlm-chat --version chat  --english --bf16
torchrun --standalone --nnodes=1 --nproc-per-node=2 webdb.py --from_pretrained /ML-A100/sshare-app/saiwanming/models/cogvlm-chat --version chat --english --bf16