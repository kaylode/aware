# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
ARG BASE_IMAGE=ubuntu:18.04

# Instal basic utilities
FROM ${BASE_IMAGE} as dev-base
RUN  apt-get clean && apt-get update && apt-get upgrade && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    gcc \
    wget \
    libjpeg-dev \
    zip \
    swig python3-dev \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda-installs
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.3
ARG PYTORCH_VERSION=1.12.1
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN curl -fsSL -v -o ~/mambaforge.sh -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
    chmod +x ~/mambaforge.sh && \
    ~/mambaforge.sh -b -p /opt/mamba && \
    rm ~/mambaforge.sh && \
    /opt/mamba/bin/mamba install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y \
    python=${PYTHON_VERSION} \
    pytorch=${PYTORCH_VERSION} torchvision "cudatoolkit=${CUDA_VERSION}" && \
    /opt/mamba/bin/mamba clean -ya

# Install repo
RUN --mount=type=secret,id=gh_token \
  GIT_AUTH_TOKEN=$(cat /run/secrets/gh_token) \
  && git clone "https://kaylode:$GIT_AUTH_TOKEN@github.com/kaylode/hipe-explore.git" /workspace/hipe/
WORKDIR /workspace/hipe

# Install requirements
RUN /opt/mamba/bin/python -m pip install -r requirements.txt
RUN /opt/mamba/bin/python -m pip install jupyter nbconvert

ENV PATH /opt/mamba/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

WORKDIR /workspace/hipe
RUN mkdir /workspace/hipe/outputs/

ENTRYPOINT ["/bin/sh"]