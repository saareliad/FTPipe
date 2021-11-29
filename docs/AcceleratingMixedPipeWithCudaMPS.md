### Accelerating mixed pipe with CUDA MPS
As $UID, run the following commands
```
ulimit -n 16384

# export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that’s
accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that’s
accessible to the given $UID
nvidia-cuda-mps-control -d # Start deamon in background
```

To shutdown:
```bash
echo quit | nvidia-cuda-mps-control