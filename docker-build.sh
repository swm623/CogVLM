export IAMGE_TAG=ccr-276x7ilk-vpc.cnc.bj.baidubce.com/ai/cogvlm:cuda11.7-wandb

docker build -t $IAMGE_TAG . -f Dockerfile

docker save $IAMGE_TAG -o cogvlm.tar

bcecmd bos cp -y cogvlm.tar bos://bj-ai-data/sshare-app/docker-images/cogvlm.tar