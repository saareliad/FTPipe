# Simplest example

## Environment
(Experimental, unchecked) env without MPI.
```bash
conda env create -f env_utils/env_without_mpi.yml
conda activate nompi
```

(if you see import errors just install the missing packages.)

## Data
use
```bash
python download/datasets/download_datasets.py
```
to get several datasets.

or execute just the relevant part in python
```python
from torchvision.datasets import CIFAR10
DATA_DIR='/home_local/saareliad/data' # replace something of yours ;
CIFAR10(root=DATA_DIR, download=True, train=True) 
CIFAR10(root=DATA_DIR, download=True, train=False)
```

## Run
(single machine optimized streams, no MPI build needed)
```
python main.py –mode mp –config configs/cv/cifar10/wrn_16x4_c10_p2/stale_nr.json –seed 42
```

### Optional: single GPU

Can change to make it run pipeline on single GPU by changing 
relevant lines in [configs/cv/cifar10/wrn_16x4_c10_p2/stale_nr.json](configs/cv/cifar10/wrn_16x4_c10_p2/stale_nr.json)\
to
```json
    "stage_to_device_map": [0, 0],
    "nprocs": 1,
```

## Model
Simplest PoC partitioning.\
Auto-generated: [models/partitioned/wrn_16x4_c10_p2.py](models/partitioned/wrn_16x4_c10_p2.py)
Code handling reading the config is mostly here:
[models/simple_partitioning_config.py](models/simple_partitioning_config.py)