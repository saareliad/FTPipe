
# FTPipe

This repository contains code used for FTPipe USENIX ATC21 [paper](https://www.usenix.org/system/files/atc21-eliad.pdf) "Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism", and future works.

See [citation](#citation) information at the bottom of this readme.


## Overview
This repository was used to explore various unexplored territories of pipeline-model-parallelism.
It is capable of automatically partitionining, training and fine-tuning giant neural networks, with both synchronous and asynchronous pipelines. \
Code for Pipeline Staleness Mitigation study is included as well.


Models supported and tested are Huggingface transformers  (T5, GPT2, BERT, RoBerta...), many Torchvision models (probably all), and Vision Transformers. (conducted an out-of-the-box ViT PoC with the first pytorch implementation, by timm, right when it apeared.)\
The setup for T5-11B is currently kept on a seperate branch.

## Basic Usage

Clone the repository:
```
git clone https://github.com/saareliad/FTPipe.git
```
All code is currently designed to run from repository root.

After completing the [environment setup](#setup), FTPipe's usage is mainly the two following steps:
1. Partitioning models
2. Running models.

```bash
python -m autopipe.partition ... # partition models
```

```bash
python -m pipe.main ... # train models (+eval)
```

Additional documentations:
* Training arguments should be passed via json [configuration files](https://github.com/saareliad/FTPipe/blob/master/pipe/configs) (*)
* New Models, training/fine-tuning tasks, and datasets should be [registered](docs\NewModels.md) to the framework.
* Additional arguments are passed as cmd args. Do use the `--help` option to exlore. (NOTE: It is also possible to override some configuration arguments using the command line, use with caution. Partitioning uses mostly cmd args.)
* As P2P communication is done with MPI, running models often looks like this

```bash
mpirun -np 8 python -m pipe.main --config $PATH_TO_JSON_CONFIG
```
* Refer to [examples](https://github.com/saareliad/FTPipe/tree/master/t5_used_scripts_example) of recent scripts we used to partition and conduct T5 experiments. 
* Do feel free to contact (issue/mail/linkedin/...).

(*Note: a more comprehensive explanation is planned, meanwhile, configuration can be understood via examples or code).

## Setup

* Follow the [instructions](pipe/env_utils/create_env_new_server_new.sh) to setup the required conda env. This includes building pytorch from source with cuda-aware openmpi.
* NOTE: Model partitioning can be done using a [much simpler conda env](https://github.com/saareliad/FTPipe/blob/main/pipe/env_utils/env_without_mpi.yml) (without mpi or building from source)
```
conda env create -f pipe/env_utils/env_without_mpi.yml
```


The simiple recpie below was used to set it up on our servers
```bash
BUILD_DIR=<SOMEPLACE_FOR_DOWNLOADED_SOFTWARE> # openmpi, pytorch
cd pipe/env_utils
cp create_env_new_server.sh $BUILD_DIR
cd $BUILD_DIR
vim create_env_new_server.sh  # change paths: home_local, FTPIPE_ROOT
bash create_env_new_server.sh # it is safer to run it step by step.
```
where `$BUILD_DIR` is set to a a repository to place the clones of openmpi and pytorch.


### Aditional docs
Work in progress to add all docs in thier own [docs directory](docs/).

Some additional usage instructions are documented across the repository.
For example: 
 - At the [pipe](pipe/) module, there are instructions and scripts for running downloading data, 
 - Refer to the [pipes-list](docs\PipeList.md) for availalble staleness mitigation and pipelines which can be used at runtime.
 - See the [autopipe](autopipe/) module for avaialbe partitioning methods. See the [tasks](autopipe/tasks) directory for examples of partitioning tasks (e.g., differnt models architechtures or downstream fine-tuning tasks). 
 - A detailed example of steps/changes taken to export a T5 model from huggingface can be found [here](models/new_t5_example).

## Note
_Note: some hyper-parameters in mpipe partitioning (e.g., GPU memory capacity), env and so on are still hardcoded to our and not available as cmd options. Currently, one will need to change them change them manually to experiment (As we did...)_

## Citation
```
@inproceedings {ftpipe,
author = {Saar Eliad and Ido Hakimi and Alon De Jagger and Mark Silberstein and Assaf Schuster},
title = {Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism},
booktitle = {2021 {USENIX} Annual Technical Conference ({USENIX} {ATC} 21)},
year = {2021},
isbn = {978-1-939133-23-6},
pages = {381--396},
url = {https://www.usenix.org/conference/atc21/presentation/eliad},
publisher = {{USENIX} Association},
month = jul,
}
```
