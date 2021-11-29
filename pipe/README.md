# FTPipe

Pipeline Runtime

```bash
python -m pipe.main ... # train models (+eval) (+preprocess)
```

Do use the `--help` and examples to explore.

## Get the data
```bash
python pipe/data/download/datasets/download_datasets.py
```
Data for T5 tasks is obtained by using `--mode perprocess`
cmd option.
## Run

### Choose a config
See [configs](configs/) for config examples.

To choose a spesific config, add it to command line:

```bash
mpirun -np 2 python -m pipe.main --config $PATH_TO_CONFIG
```
without doing so, it will run the [dummy config](configs/dummy.json) (created for dev usage).

### Preprocess
if data preprocessing is needed, run the selected config with:
```bash
python -m pipe.main --mode preproc --config $PATH_TO_CONFIG ...
```

### MPI

cuda aware openmpi:

```bash
mpirun -np 2 python -m pipe.main --config $PATH_TO_CONFIG
```

### Multiprocessing
A PoC runtime which can be used with the `--mode mp` cmd option.

Supposed to work for very simple stright pipelines only (mostly- torchvision models, vit), and is BUGGY when configuration gets more exotic. (e.g, Tied Wieghts)
```bash
python -m pipe.main --nprocs 2 --mode mp --config $PATH_TO_CONFIG
```
