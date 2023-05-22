# README
# https://github.com/NVIDIA/nvidia-docker
# https://forum.manjaro.org/t/howto-installing-docker-and-nvidia-runtime-my-experience-and-howto/97017
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide
# https://github.com/Lightning-AI/lightning/tree/master/dockers

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0
ARG CUDA_VERSION=11.7.1

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}
#FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

WORKDIR /app

# To copy everythin in one layer:
#COPY nianetcae /app/nianetcae

# To copy in multiple layers:
COPY configs /app/configs
COPY data /app/data
COPY nianetcae/dataloaders /app/nianetcae/dataloaders
COPY nianetcae/experiments /app/nianetcae/experiments
COPY nianetcae/models /app/nianetcae/models
COPY nianetcae/niapy_extension /app/nianetcae/niapy_extension
COPY nianetcae/storage /app/nianetcae/storage
COPY nianetcae/visualize /app/nianetcae/visualize

COPY nianetcae/__init__.py /app/nianetcae/__init__.py
COPY nianetcae/cae_run.py /app/nianetcae/cae_run.py

COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY main.py /app/main.py

RUN pip3 install .
RUN pip3 install -r requirements.txt
RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info
#CMD [ "python" , "main.py"]