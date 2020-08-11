ARG baseImage=nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM $baseImage
# pytorch args
# ARG PYTHON_VERSION=3.7
# ARG WITH_TORCHVISION=1

# building ompi
RUN apt-get update && \
    apt-get install -y --no-install-recommends  \
        bzip2 \
        ca-certificates \
        curl \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O /opt/miniconda.sh && \
    chmod +x /opt/miniconda.sh && \
    /opt/miniconda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda update -n base conda && \
    rm /opt/miniconda.sh

ENV PATH /opt/conda/bin:${PATH}

RUN conda install -c anaconda \
        anaconda-client \
        conda-build \
        conda-verify && \
    conda clean -ya

COPY . /opt/ompi-recipe

WORKDIR /opt/ompi-recipe
