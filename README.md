
# FTPipe

This repository contains code used for FTPipe USENIX ATC21 paper and future works. (Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism.

An arxiv version and citation are coming soon.

Code for Pipeline Staleness Mitigation is also included.

## Overview
This repository is used to explore various methods of pipeline-model-parallelism.
It is capable of automatically partitionining, training and fine-tuning giant neural networks, with both synchronous and asynchronous pipelines.


## Usage

Note: The full readme is still WIP and TBD. Howver there is a partial recipe below and some [examples](https://github.com/saareliad/FTPipe/tree/master/prepare_new_t5) of recent scripts we used.  Feel free to contact via email.

0. clone the repository.
1. To run partitioning, prepare a `Task`, which is simply a model and example inputs for it. Place it [here](autopipe/tasks).

2. Choose partitioning and analysis settings, for example:
    ```bash
    python -m autopipe.partition vision --crop 32 --no_recomputation -b 256 -p 4 --save_memory_mode --partitioning_method pipedream --model wrn_28x10_ c100_dr03_gn
    ```
    This will create, compile, and autogenerate the partitioned model and place it [here](models/partitioned).
    
    _Note: some hyper-parameters in mpipe partitioning, env and so on are still hardcoded and not available as cmd options._

3. Register the partitioned model to the pipeline runtime. 

    In our experiments, this is done by implementing a `CommonModelHandler`, which handles this logic 
    ([see examples](pipe/models/registery)). Note that some models may require additional settings.

4. Register a new dataset or use an existing one.  

    In our experiments, this is done by implementing a `CommonDatasetHandler`.
    The logic for doing so is [here](pipe/data).
    
    Note that additional logic is added to prevent unnecessary data movements automatically.

5. Define training and staleness mitigation settings. Then, run experiments with desired settings.

    In our experiments, this is done by passing a json configuration ([examples](pipe/configs)).
   An example run:
    ```bash
   python -m pipe.main --config pipe/configs/cv/cifar100/wrn28x10/no_recomputation/stale_nr.json --bs_train_from_cmd --bs_train 16 --step_every_from_cmd --step_every 16 --seed 42 --mode mp
   ```    
    Finally, a json file with results will be created and placed as defined in the config.
    
## Accelerating mixed pipe with MPS
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
```
