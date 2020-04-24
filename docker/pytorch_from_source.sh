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


# cudnn: login, start and stop download in chrome, goto downloads, copy link, 
# and then wget it. e.g:
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/Ubuntu18_04-x64/libcudnn7_7.6.5.32-1%2Bcuda10.2_amd64.deb?XuvLN92s-qrIUfqyCQKHhau-aJJ37MXoH0YFXF_xob58w5ecCAJSJcoC_6hYQDyBooQ9avrDoq_VuCVPP3-chThGq0am034OYtRmQxCqMW_lAzC4Ns0zv6iwQSE2SJ9qQka0sl6xG_9mlYICKFHm5T1psr-vSBGd5TdA3Gj8Vi5g48hUC-sbu0ghfZoZJUy_QErNHF2wYqAD115YB4pvoHnlmK2CxSzKuWwnP00Y4ehQ40MmIg2k84_WHftpnmt- \ 
-O libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb

# check we have nvcc
which nvcc

# needed:
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
conda install -c saareliad se-msnag2  # This is important.
conda install -c pytorch magma-cuda102

# Get pytorch source
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# Added required version
git checkout v1.5.0
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# Install
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
