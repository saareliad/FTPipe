
# PYTORCH_LOCATION=/home/alonde-jager/gpipe-research/pytorch
PYTORCH_LOCATION=/home_local/saareliad/pytorch

git clone https://github.com/pytorch/pytorch.git $PYTORCH_LOCATION
cd $PYTORCH_LOCATION; git checkout v1.4.0 ; cd -

rsync pytorch/ ${PYTORCH_LOCATION}/ -ravh
cd $PYTORCH_LOCATION
docker build -t pytorch -f docker/pytorch/Dockerfile .
cd -

