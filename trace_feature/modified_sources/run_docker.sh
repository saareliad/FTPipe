# Run from main repo:
# may change image name.
# TODO: share cache for models/datasets
docker run -it --ipc=host --rm -v $(pwd):/workspace partitioning/live
