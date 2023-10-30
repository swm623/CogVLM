FROM diffusers-pytorch-cuda:11.7.1-cudnn8-runtime-ubuntu20.04
LABEL maintainer="01.ai"
LABEL repository="sd-text_to_image"

WORKDIR /app

RUN echo "deb http://archive.ubuntu.com/ubuntu/ focal main restricted" >> /etc/apt/sources.list && \
    apt update && \
    apt -y install libibverbs1 \
    ibverbs-utils \
    libibverbs-dev \
    make &&\
    rm -rf /var/lib/apt/lists

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

