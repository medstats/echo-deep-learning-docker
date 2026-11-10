FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    pillow \
    matplotlib \
    scikit-learn \
    scikit-image \
    torchvision \
    pyyaml \
    timm \
    open_clip_torch \
    ftfy \
    opencv-python-headless \
    pydicom \
    transformers==4.57.0

COPY repos/ /workspace/repos/
COPY strain/ /workspace/strain/
COPY *.py /workspace/
COPY weights/manifest.json /workspace/weights/manifest.json
COPY weights/strain/manifest.json /workspace/weights/strain/manifest.json

# Pre-download all 5 local echo model weights
RUN mkdir -p /workspace/weights/echoclip /workspace/weights/strain && \
    curl -L -s -o /workspace/weights/echonet.pt https://github.com/echonet/dynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt && \
    curl -L -s -o /workspace/weights/panecho.pt https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt && \
    curl -L -s -o /workspace/weights/echoclip/open_clip_pytorch_model.bin https://huggingface.co/mkaichristensen/echo-clip/resolve/main/open_clip_pytorch_model.bin && \
    curl -L -s -o /workspace/weights/strain/unity-211-weights-401.pt https://data.unityimaging.net/buckets/matt-output/checkpoints/unity/unity-211/weights-401.pt && \
    curl -L -s -o /workspace/weights/strain/deeplabv3_resnet50_random.pt https://github.com/echonet/dynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt

# Pre-download EchoPrime core encoder model data
RUN mkdir -p /workspace/cache/echoprime && \
    (curl -L -s -o /workspace/cache/echoprime/model_data.zip https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip && \
     unzip -q /workspace/cache/echoprime/model_data.zip -d /workspace/repos/EchoPrime/ && \
     rm -f /workspace/cache/echoprime/model_data.zip) || true

CMD ["python", "/workspace/run_inference.py", "--help"]
