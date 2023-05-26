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

# https://stackoverflow.com/a/54763270/6152465
ARG YOUR_ENV

ENV YOUR_ENV=${YOUR_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# Copy the poetry.lock and pyproject.toml files to the working directory
COPY poetry.lock pyproject.toml /app/

# Install Poetry
RUN pip install poetry

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi && \
    poetry run poe autoinstall-torch-cuda

# Creating folders, and files for a project:
COPY data /app/data

COPY nianetcae/niapy_extension /app/nianetcae/niapy_extension
COPY nianetcae/storage /app/nianetcae/storage
COPY nianetcae/visualize /app/nianetcae/visualize
COPY nianetcae/dataloaders /app/nianetcae/dataloaders
COPY nianetcae/experiments /app/nianetcae/experiments
COPY nianetcae/models /app/nianetcae/models

COPY nianetcae/__init__.py /app/nianetcae/__init__.py
COPY nianetcae/cae_run.py /app/nianetcae/cae_run.py

COPY tests/test_model.py /app/tests/test_model.py
COPY tests/__init__.py /app/tests/__init__.py

COPY configs /app/configs
COPY setup.py /app/setup.py
COPY main.py /app/main.py

#OLD

#ARG PYTHON_VERSION=3.10
#ARG PYTORCH_VERSION=2.0
#ARG CUDA_VERSION=11.7.1
#
#FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}
##FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1
#
#LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"
#
#WORKDIR /app
#
## To copy everythin in one layer:
## Rebuilding docker image will be slower on change, but the image will be smaller
##COPY nianetcae /app/nianetcae
##COPY tests /app/tests
#
## To copy in multiple layers:
## Rebuilding docker image will be faster on change, but the image will be bigger
#COPY requirements.txt /app/requirements.txt
#COPY data /app/data
#
#RUN pip3 install .
#RUN pip3 install -r requirements.txt
#RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info
#
#COPY nianetcae/niapy_extension /app/nianetcae/niapy_extension
#COPY nianetcae/storage /app/nianetcae/storage
#COPY nianetcae/visualize /app/nianetcae/visualize
#COPY nianetcae/dataloaders /app/nianetcae/dataloaders
#COPY nianetcae/experiments /app/nianetcae/experiments
#COPY nianetcae/models /app/nianetcae/models
#
#COPY nianetcae/__init__.py /app/nianetcae/__init__.py
#COPY nianetcae/cae_run.py /app/nianetcae/cae_run.py
#
#COPY tests/test_model.py /app/tests/test_model.py
#COPY tests/__init__.py /app/tests/__init__.py
#
#COPY configs /app/configs
#COPY setup.py /app/setup.py
#COPY main.py /app/main.py
##CMD [ "python" , "main.py"]