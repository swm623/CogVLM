FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
LABEL maintainer="01.ai"
LABEL repository="sd-text_to_image"

WORKDIR /app


COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117  &&\
    python3 -m pip install --no-cache-dir -r requirements.txt

