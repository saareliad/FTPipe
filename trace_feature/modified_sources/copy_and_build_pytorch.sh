
# PYTORCH_LOCATION=/home/alonde-jager/gpipe-research/pytorch
PYTORCH_LOCATION=/home_local/saareliad/pytorch

if [[ ! -d "$PYTORCH_LOCATION" ]]; then
git clone https://github.com/pytorch/pytorch.git $PYTORCH_LOCATION
cd $PYTORCH_LOCATION; git checkout v1.4.0 ; cd -
git submodule update --init --recursive
fi

# git apply my_awesome_change.patch
rsync pytorch/ ${PYTORCH_LOCATION}/ -ravh
echo "-I- APPLIED OUR PATCH"


cd $PYTORCH_LOCATION
docker build -t pytorch:trace_feature -f docker/pytorch/Dockerfile .

