# Run from main repo:
# may change image name.
# TODO: share cache for models/datasets
# docker run -it --ipc=host --rm -v $(pwd):/workspace partitioning/live
docker run -it --ipc=host --rm -p 12345:12345 -v $(pwd):/workspace:Z partitioning/live
