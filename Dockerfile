# README
# https://github.com/NVIDIA/nvidia-docker
# https://forum.manjaro.org/t/howto-installing-docker-and-nvidia-runtime-my-experience-and-howto/97017
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide
# https://github.com/Lightning-AI/lightning/tree/master/dockers

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0
ARG CUDA_VERSION=11.7.1
ARG LIGHTNING_VERSION=""

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}
#FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"


WORKDIR /app

COPY configs configs
COPY data data
COPY dataloaders dataloaders
COPY experiments experiments
COPY models models
COPY niapy_extension niapy_extension
COPY storage storage

COPY requirements.txt requirements.txt
COPY cae_run.py /app

RUN pip3 install -r requirements.txt
RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info

