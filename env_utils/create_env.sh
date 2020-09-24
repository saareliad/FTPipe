# Make sure you have prerequisites, e,g:
#
# apt-get install -y --no-install-recommends \
#          build-essential \
#          cmake \
#          git \
#          curl \
#          ca-certificates \
#          libjpeg-dev \
#          libpng-dev

# Also needed:
# cuda 10.2
# cudnn (NOTE: install from tar...)

conda_stuff() {
    CONDA_BASE=$(conda info --base)
    source ${CONDA_BASE}/etc/profile.d/conda.sh
}

new_env() {
  # conda create -y -n pt python=3.8 numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses cython
    conda create -y -n py38 python=3.8
    conda activate py38
    conda install -y numpy
    conda install -y pyyaml
    conda install -y scipy
    conda install -y ipython
    conda install -y mkl
    conda install -y mkl-include
    conda install -y ninja
    conda install -y cython
    conda install -y -c pytorch magma-cuda102
    conda install -y cffi
}

avx2pillow_torchvision() {
    # Install torchvision:
    # Note: we need to do it after we installed pytorch.
    # (1) Install faster pollow-simd with AVX2 support.
    # can if we have AVX2 support with grep avx2 /proc/cpuinfo
    pip uninstall pillow
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
    # (2) Install torchvision from source.
    pip install git+https://github.com/pytorch/vision.git@v0.6.0
}

cuda_awre_openmpi() {
    # Can also do it from conda, but I didn't try.
    sudo apt-get install libnuma-dev
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz
    tar -xvzf openmpi-4.0.5.tar.gz
    cd openmpi-4.0.5 || exit 1
    ./configure --with-cuda
    make
    sudo make install
    sudo ldconfig
}

pytorch_sources() {
    # NOTE: The recursive is important
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch || exit 1
    # Added required version (not checked the line below)

    git checkout --recurse-submodules v1.6.0

    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive
}

pytorch_install() {
    # this is for RTX2080 (Turing).
    # select your GPUs:
    #
    # 	named_arches = collections.OrderedDict([
    # 	('Kepler+Tesla', '3.7'),
    # 	('Kepler', '3.5+PTX'),
    # 	('Maxwell+Tegra', '5.3'),
    # 	('Maxwell', '5.0;5.2+PTX'),
    # 	('Pascal', '6.0;6.1+PTX'),
    # 	('Volta', '7.0+PTX'),
    # 	('Turing', '7.5+PTX'),
    # ])
    # NOTE: must avoid conda env polution.
    export TORCH_CUDA_ARCH_LIST="7.5+PTX"
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 $CFLAGS"
    python setup.py install
}

conda_stuff
new_env
cuda_awre_openmpi
pytorch_sources
pytorch_install
avx2pillow_torchvision
