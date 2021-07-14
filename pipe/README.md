# FTPipe
# Asynchronous Pipeline

## Available algorithms

### Stale-synchronous pipeline

- recomputation / no recomputation
- stale
- weight stashing (ws)
- weight prediction (wp) : msnag, aggmsnag

  - sgd
  - adam
  - adamw

- "scheduler aware prediction"

- gap aware (ga)

  - sgd, adam, adamw

- combinations {wp, ws, ga}
- gradient aggregation in pipeline (`step_every`)

Note that there is difference between stale and weight prediction with recomputation and without.

Weight predicion is often called `msnag` in code.

### Fully-synchronous

- gpipe
- DistributedDataParallel (DDP): SSGD
- Sequential (seq): naive inter-layer model parallelisem (multi gpu)
- single gpu for small model.

## Setup env
There are several option to do so. here is the main one.

### From source (new)

Follow instruction [here](env_utils/create_env_new_server.sh), then run it to build pytorch from source with cuda-aware openmpi.

```bash
cd env_utils
cp create_env_new_server.sh $BUILD_DIR
cd $BUILD_DIR
vim create_env_new_server.sh
# after editing...
bash create_env_new_server.sh
```

## Get data
```bash
python download/datasets/download_datasets.py
```

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

```bash
python -m pipe.main --nprocs 2 --mode mp --config $PATH_TO_CONFIG
```

## Pitfalls

- **_recomputation and normalization layers_** For some normalization layers (e.g BatchNormalization) the recomputation does two forward passes (dummy, and during recomputation), therefore we monkey patch it during the dummy pass. Can do with with re-writing the layer too, which is slightly more efficient than monkey patching.

- **_recomputation: uses a correct random seed_** We use the same random same seed for the two recomputation forward passes.

- **_Sending large tensors with MPI_** takes extra memory due buffer allocations.

- **_NLP models_** check you overwrite cash.

### Other Known challenges

1. MPI: 
  - we need to pop/destroy async handlers, otherwise memory explodes. We handle this in python.
  (may change cpp code in torch mpigroup to drop the tensor and its pointers once ISends are completed, so we won't have to handle mem cleaning).
  - must explicitly do `cuda.synchronize(...)` before CUDA-AWARE MPI sends.

2. Multiprocessing mode: (`--mode mp`) requires multiple cuda contexts per GPU. It is sometimes not robust since it was not used a lot. It works moslty for simple vision models. I did not invest it aditional time since better communication libraries are being developed.

3. Tied weights: MPI: large memory consumption for very distant sends. Multiprocessing/cudaIPC:(using the same tensor): may be a race condition. wieght prediction: we change and send the weight itself (this has advantage and disadvantage).

4. `in_place` operations (like ReLU) at partition border have a potential of destroying our saved activations (also throws some autograd "variable changed inplace" error) => solution: automatically replace inplaces ops on partition borders. (the solution worked for resnets, will not work for more complex models). IMO this should be solved by partitioning.

5. In Pipedream they pass the target all across the pipeline (waste of energy, inefficient). solution: using two data-loaders with correct synchronization.


## TODOs

- Memory efficient Gap Aware for entire pipeline  (+`step_every`)
  (When delay is 1, we can do gap aware even on deeper partitions without stashing)

- weight prediction: currenrlty, we do some extra "reverts" (e.g in case of several backwards one after another and weight stashing) Check this. its very small optimization (not in steady state), and may be yucky to implement.

- evaluation/statistics by number of steps and not just epochs. (currenrlty- steps are automaticlly translated to epochs)

- fix batch normalization in `torch.no_grad()` to something better than monkey patch


## See For MPI:

- [running with mpi](https://www.open-mpi.org/faq/?category=running) especially see [mpi-env-vars](https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables).

## Debugging

- [debugging mpi python applications with vscode](https://gist.github.com/asroy/ca018117e5dbbf53569b696a8c89204f)

  - debug work only when dataloading is on main thread. (`num_data_workers=0`).
  - run same thing, with `--debug` flag, then wait for attachment:

  > > ```bash
  > > mpirun -np 2 python main.py --debug <LIST OF RANKS>
  > > ```

- If you debug cuda, you may want to fix the trace by:

  > > ```bash
  > > CUDA_LAUNCH_BLOCKING=1 mpirun -np 2 python main.py --debug <LIST OF RANKS>
  > > ```

- Before you debug, you may want to check run the error is cuda specific and not cpu

## Misc

- Communication Matrix Embedding with cuda P2P samples (15% BW improvment for pipeline). Can use this [script](/misc/p2p_bw_mat.sh).

- To see how pytorch is compiled, use
```
torch.__config__.show()

- use node 0 : `numactl --cpunodebind=0` (sudo apt install numactl)
  - checking this: either `lstopo` or `lspci -vvv | less`.

#  0.6.4 T5
```
