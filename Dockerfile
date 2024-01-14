# Dockerfile for OmniSafe
#
#   $ docker build --target base --tag omnisafe:latest .
#
# or
#
#   $ docker build --target devel --tag omnisafe-devel:latest .
#

ARG cuda_docker_tag="11.7.1-cudnn8-devel-ubuntu22.04"
FROM nvidia/cuda:"${cuda_docker_tag}" AS builder

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Install packages
RUN apt-get update && \
    apt-get install -y sudo ca-certificates openssl \
        git ssh build-essential gcc g++ cmake make \
        python3-dev python3-venv python3-opengl libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV MUJOCO_GL osmesa
ENV PYOPENGL_PLATFORM osmesa
ENV CC=gcc CXX=g++

# Add a new user
RUN useradd -m -s /bin/bash omnisafe && \
    echo "omnisafe ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER omnisafe
RUN echo "export PS1='[\[\e[1;33m\]\u\[\e[0m\]:\[\e[1;35m\]\w\[\e[0m\]]\$ '" >> ~/.bashrc

# Setup virtual environment
RUN /usr/bin/python3 -m venv --upgrade-deps ~/venv && rm -rf ~/.pip/cache
RUN PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu$(echo "${CUDA_VERSION}" | cut -d'.' -f-2  | tr -d '.')" && \
    echo "export PIP_EXTRA_INDEX_URL='${PIP_EXTRA_INDEX_URL}'" >> ~/venv/bin/activate && \
    echo "source /home/omnisafe/venv/bin/activate" >> ~/.bashrc

# Install dependencies
WORKDIR /home/omnisafe/omnisafe
COPY --chown=omnisafe requirements.txt requirements.txt
RUN source ~/venv/bin/activate && \
    python -m pip install -r requirements.txt && \
    rm -rf ~/.pip/cache ~/.cache/pip

####################################################################################################

FROM builder AS devel-builder

# Install extra dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y golang && \
    sudo chown -R "$(whoami):$(whoami)" "$(realpath /usr/lib/go)" && \
    sudo rm -rf /var/lib/apt/lists/*

# Install addlicense
ENV GOROOT="/usr/lib/go"
ENV GOBIN="${GOROOT}/bin"
ENV PATH="${GOBIN}:${PATH}"
RUN go install github.com/google/addlicense@latest

# Install extra PyPI dependencies
COPY --chown=omnisafe tests/requirements.txt tests/requirements.txt
RUN source ~/venv/bin/activate && \
    python -m pip install -r tests/requirements.txt && \
    rm -rf ~/.pip/cache ~/.cache/pip

####################################################################################################

FROM builder AS base

COPY --chown=omnisafe . .

# Install omnisafe
RUN source ~/venv/bin/activate && \
    make install-editable && \
    rm -rf .eggs *.egg-info ~/.pip/cache ~/.cache/pip

ENTRYPOINT [ "/bin/bash", "--login" ]

####################################################################################################

FROM devel-builder AS devel

COPY --from=base /home/omnisafe/omnisafe .
