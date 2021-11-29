# Follow these instruction from FTPipe root dir.

# TODO: install cuda-10.2, if not already installed
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
#sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
#sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
#sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
#sudo apt-get update
#sudo apt-get -y install cuda-10-2

# TODO: install CUDNN,  if not already installed
# Go to https://developer.nvidia.com/compute/machine-learning/cudnn and download cudnn
# e.g
# https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/10.2_20201106/cudnn-10.2-linux-x64-v8.0.5.39.tgz
# tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
# sudo cp --preserve=links cuda/include/cudnn*.h /usr/local/cuda/include
# sudo cp --preserve=links cuda/lib64/libcudnn* /usr/local/cuda/lib64
# sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
# sudo ldconfig

# https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#ubuntu-lts
# sudo apt-get install cuda-drivers-440 -f

# local dir
DIR_NAME=$USER
USER_NAME=$USER
cd /home_local/
sudo mkdir $DIR_NAME
sudo chown $USER_NAME: $DIR_NAME
sudo chmod u+w $DIR_NAME

# conda
DIR_NAME=/home_local/${USER}
cd $DIR_NAME
ADDR=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
curl -o ./miniconda.sh -O ${ADDR} && \
chmod +x ./miniconda.sh # &&
./miniconda.sh -b -p ${DIR_NAME}/miniconda3

# system
sudo apt-get install -y --no-install-recommends \
build-essential \
cmake \
git \
curl \
ca-certificates \
libjpeg-dev \
libpng-dev
sudo apt-get install -y libnuma-dev

wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
tar -xvzf openmpi-4.1.1.tar.gz
cd openmpi-4.1.1 || exit 1
./configure --with-cuda \
--disable-dependency-tracking \
--disable-mpi-fortran
make -j 40
sudo make install -j 40
sudo ldconfig
cd -

conda create -y -n pt2 python=3.8 numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses cython
conda activate pt2
conda install -y -c pytorch magma-cuda102

git clone --recursive https://github.com/pytorch/pytorch pytorch_v1.8.1
cd pytorch_v1.8.1 || exit 1
git checkout --recurse-submodules v1.8.1
git submodule sync
git submodule update --init --recursive

export TORCH_CUDA_ARCH_LIST="7.5+PTX"
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export BUILD_TEST=0
export USE_IBVERBS=0
export USE_CUDNN=1
python setup.py install
cd -

FTPIPE_ROOT=~/workspace/FTPipe/
 cd ${FTPIPE_ROOT}
conda env update -f pipe/env_utils/env_add_to_build_from_source.yml

pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
# (2) Install torchvision from source.
pip install git+https://github.com/pytorch/vision.git@v0.9.1
# '4.1.1'