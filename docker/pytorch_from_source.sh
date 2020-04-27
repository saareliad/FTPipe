# Install cuda 10.2
# either runfile
# wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
# sudo sh cuda_10.2.89_440.33.01_linux.run

# or deb (network) (recommanded)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# and follow :
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
# add to ~/.bashrc
# export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64\
#                                  ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}



# check we have nvcc
which nvcc

#########
# Cudnn
#########

# Option (1): cudnn deb file. (does not work with default setup configurations)
# cudnn: login, start and stop download in chrome, goto downloads, copy link, 
# and then wget it. e.g:
# wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/Ubuntu18_04-x64/libcudnn7_7.6.5.32-1%2Bcuda10.2_amd64.deb?XuvLN92s-qrIUfqyCQKHhau-aJJ37MXoH0YFXF_xob58w5ecCAJSJcoC_6hYQDyBooQ9avrDoq_VuCVPP3-chThGq0am034OYtRmQxCqMW_lAzC4Ns0zv6iwQSE2SJ9qQka0sl6xG_9mlYICKFHm5T1psr-vSBGd5TdA3Gj8Vi5g48hUC-sbu0ghfZoZJUy_QErNHF2wYqAD115YB4pvoHnlmK2CxSzKuWwnP00Y4ehQ40MmIg2k84_WHftpnmt- \ 
# -O libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
# dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb

# Option (2): tar file.
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/cudnn-10.2-linux-x64-v7.6.5.32.tgz?M5aYUm0N4iOKhZlwo44cYGY7JrtxJ049tBS6gDo1icDc5VEHc_gj3nc4uB5A0urL7WPjmZNXscDazrAVgqUq0KTDErXgIQexFugIR8vMEhoAUeyNAtKI5FWUPoWF49MUX-1auynFmUd3dvIB0MZoNL7JoBLXON9aJWXpsrfAK-j2Qx3PPqSHxprH1gxlNlYbXcHNGfFOw9r8F2eeCC7IG-gVbq3oCZMjvg -O cudnn-10.2-linux-x64-v7.6.5.32.tgz
# see https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
 tar -xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgz

sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/include/cudnn_version.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
# NOTE: it may cause :
# https://askubuntu.com/questions/1025928/why-do-i-get-sbin-ldconfig-real-usr-local-cuda-lib64-libcudnn-so-7-is-not-a

# needed:
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
conda install -c saareliad se-msnag2  # This is important.
conda install -c pytorch magma-cuda102

# Get pytorch source
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# Added required version
git checkout --recurse-submodules v1.5.0
# if you are updating an existing checkout

git submodule sync
git submodule update --init --recursive
echo "hi"

# see: https://github.com/pytorch/pytorch/issues/9310
# https://stackoverflow.com/questions/31948521/building-error-using-cmake-cannot-find-lpthreads

# apt -y install libboost-tools-dev libboost-thread1.62-dev magics++

# https://github.com/pytorch/pytorch/issues/16112

# Install
export INTEL_MKL_DIR=$CONDA_PREFIX
export INTEL_OMP_DIR=$CONDA_PREFIX
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}


export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
/lib/x86_64-linux-gnu/


python setup.py install
