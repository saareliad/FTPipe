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
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} as dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython&& \
    /opt/conda/bin/conda install -y -c saareliad se-msnag2 && \
    /opt/conda/bin/conda clean -ya

FROM dev-base as submodule-update
WORKDIR /opt/pytorch
COPY . .
RUN git submodule sync && git submodule update --init --recursive

FROM conda as build
WORKDIR /opt/pytorch
COPY --from=conda /opt/conda /opt/conda
COPY --from=submodule-update /opt/pytorch /opt/pytorch
RUN --mount=type=cache,target=/opt/ccache \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 7.5+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install

FROM conda as conda-installs
ARG INSTALL_CHANNEL=pytorch-nightly
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -y pytorch torchvision && \
    /opt/conda/bin/conda clean -ya

FROM ${BASE_IMAGE} as official
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=conda-installs /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
WORKDIR /workspace

FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=build /opt/conda /opt/conda
