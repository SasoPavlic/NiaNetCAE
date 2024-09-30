# README
# https://github.com/NVIDIA/nvidia-docker
# https://forum.manjaro.org/t/howto-installing-docker-and-nvidia-runtime-my-experience-and-howto/97017
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide
# https://github.com/Lightning-AI/lightning/tree/master/dockers

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0
ARG CUDA_VERSION=11.8.0

# Base cuda image
#FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}
# Latest image
FROM pytorchlightning/pytorch_lightning:latest-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

WORKDIR /app

# To copy everythin in one layer:
# Rebuilding docker image will be slower on change, but the image will be smaller
#COPY nianetcae /app/nianetcae
#COPY tests /app/tests

# To copy in multiple layers:
# Rebuilding docker image will be faster on change, but the image will be bigger
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

RUN mkdir data
RUN mkdir configs
#COPY configs /app/configs

COPY nianetcae/niapy_extension /app/nianetcae/niapy_extension
COPY nianetcae/storage /app/nianetcae/storage
COPY nianetcae/visualize /app/nianetcae/visualize
COPY nianetcae/dataloaders /app/nianetcae/dataloaders
COPY nianetcae/experiments /app/nianetcae/experiments
COPY nianetcae/models /app/nianetcae/models

COPY nianetcae/__init__.py /app/nianetcae/__init__.py
COPY nianetcae/cae_architecture_search.py /app/nianetcae/cae_architecture_search.py

COPY tests/test_model.py /app/tests/test_model.py
COPY tests/__init__.py /app/tests/__init__.py

COPY setup.py /app/setup.py
COPY main.py /app/main.py
COPY log.py /app/log.py
RUN pip3 install .
RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info
##CMD [ "python" , "main.py"]