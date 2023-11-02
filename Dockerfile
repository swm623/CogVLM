FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
LABEL maintainer="01.ai"
LABEL repository="sd-text_to_image"

WORKDIR /app


COPY requirements_docker.txt .

RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia && \
    conda install -y  xformers -c xformers && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements_docker.txt &&\
    python -m spacy download en_core_web_sm

