# wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
# tar -xvzf openmpi-4.0.2.tar.gz ompi

docker build -t conda:ompi-dev . && docker run --rm -it --runtime nvidia conda:ompi-dev /bin/bash
# inside-container# anaconda login
# inside-container# conda build ompi-cuda
# inside-container# anaconda upload /path/to/ompi-cuda/package.tar.gz

# # The path is determind by prefix env var (?)
# # export PREFIX=''
# Or somewhere in conda build: /opt/conda/... see dockerfile.