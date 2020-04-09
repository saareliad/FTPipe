# Multi Step NAG Asynchronous Pipeline

## Setup env

```bash
make env
```

(or run it step by step)

## Run

### Choosing a config

Note the the examples below run the `configs/dummy.json` config (for dev usage).

To choose a spesific config, add it as an option, e.g:

```bash
mpirun -np 2 python main.py --config configs/<CONFIG_NAME> 
```

### MPI

cuda aware openmpi:

```bash
mpirun -np 2 python main.py
```

on the rishon sever:

```bash
salloc -n2 --gres=gpu:2 mpirun python main.py
```

Note that OpenMPI may require special settings (e.g disable P2P via PCI ```--mca btl_smcuda_use_cuda_ipc 0```).

### Via torch.distributed.launch model

Not supported. To be removed.

* will not work for MPI backend.
* (NOTE) nccl/gloo does broadcasting which we do not support yet.

```bash
python -m torch.distributed.launch --nnodes 1 --master_port 6005 --nproc_per_node 2 main.py --cpu --distributed_backend gloo
```

## Adding a model

### Example for a transformer model
may change in the future.

1. partition the model
2. add the model under `modles.partitioned.FN`
3. (can skip this part) create a function name `FN()` in `models.transformers_cfg` defining the config.
4. add that function to `models.transformers_cfg.MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS` 
5. register it with `models.cfg_to_model._register_model` by passing a dummy dict:
 ```python
 _register_model(dict(FN=dict()), None)
 ```
## Known problems

* gpu bcast not working yet. (some deadlock)
  * The problem is that p0 does 'send' bcast to p1 while p1 does 'send' bcast to p0, on the same group. Therefore need more groups.

* pipedream scheduler stopped working (deadlock)?

* We need to pop/destroy async handlers, otherwise memory explodes.

* With `torch.distributed.launch` we may need to manually do ```kill -9 <PIDs>``` in case of errors to kill workers.

* We currently must do `drop_last=True` because we use constant buffer buffers
  (we can write code to fix that, but we prefer this easy fix meanwhile).

* the first pass on input to determine shapes: changes batch normalization statistics (can do it with monkey patch, but its super ugly!)

* must explicitly do `cuda.synchronize(...)` before CUDA-AWARE MPI sends.

* for simulation: to communicate gpu->to-same->gpu need to do [something ugly like this](http://cudamusing.blogspot.com/2013/07/enabling-cuda-multi-process-service-mps.html).

## Some Solved problems:
* problem: batch normalization statistics keep going in `torch.no_grad()` => solution: monkey patch.

* problem: `in_place` operations (like ReLU) at partition border have a potential of destroying our saved activations (also throws some autograd "variable changed inplace" error) => solution: automatically replace inplaces ops on partition borders.

* problem: In Pipedream they pass the target all across the pipeline (waste of energy, inefficient). solution: using two data-loaders with correct synchronization.
  * Note that sometimes (in LM) we don't even need to send the "y" as its efficiently created from the "x", so we support both methods.

* recomputation: overlap with wait.

* simulation with more than one partition per GPU + cuda aware: add `--mca btl_smcuda_use_cuda_ipc_same_gpu 0` to mpirun.

* for delay=0 with msnag without weight stashing, so far I did `nag_with_predictor` even for non last partitions. This is problematic, as we would like to **return to the moved weights** in the backward pass. So what we can do is, create a dict were we save staleness from fwd to backward (we already do this, but for GA purpose) and if the staleness is 0 (and not last partition, which is the case in `run_batch_backward`) then, with the weight predictor: do `setup(0)->forward()->recomputation()->backward()->revert()->step`. 
(The condition for this is weight predictor + nag with predictor, without weight stashing)

## TODOs

* Currently, we do some extra "reverts" (e.g in case of several backwards one after another and weight stashing) Check this. its very small optimization I guess (not in steady state), and may be yucky to implement.

* with `step_every` > 2 and weight stashing -> we don't really understand whats going on.

* with `step_every`=4 for example, we can save some computations for MSNAG by using more memory. but this is not so interesting.

* GBN in pipe (support it by scheduler)

* fix batch normalization in `torch.no_grad()` to something better than monkey patch
  * Monkey patch batch normalization: is just a moneky patch.
  * can write spesific class to every case (e.g Batch Norm) to make it more efficient.

* Double Buffering (Remember credit to Mark)
* timing (ms per batch, fwd, bwd, etc)
* schedule: find way to eliminate bubbles

* CUDA aware openmpi
  * test to make sure we get the desired speedup [unlike this guy](https://www.pugetsystems.com/labs/hpc/P2P-peer-to-peer-on-NVIDIA-RTX-2080Ti-vs-GTX-1080Ti-GPUs-1331/#test-setup-and-results).

* Support multi node after everything works.
* later change `batch_idx` to `micro_batch_index` to be consistent with the paper.

* Can do gap aware at step 2 even on deeper partitions without stashing (effective only once per epoch). This is effecitve when using "batch aggregation staleness mitigation". Currently we can do it without stashing when staleness is 1.
* gap aware with weight stashing
* add "scheduler aware prediction".
* can wait in the last layer (extra irecv) to overlap (more) communication with computation.
  * this requires another rcv buffer but in the last partition we don't have mem problems.

* change c code in torch mpigroup to drop the tensor (and its pointers) once completed, so we won't have to handle mem cleaning (this can reduce peak memory memory).

* option in GA to change gradient activation before sending!

## References

* [running with mpi](https://www.open-mpi.org/faq/?category=running)
  * especially see [mpi-env-vars](https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables).

## Debugging

* [debugging mpi python applications with vscode](https://gist.github.com/asroy/ca018117e5dbbf53569b696a8c89204f)
  * debug work only when dataloading is on main thread. (`num_data_workers=0`).
  * run same thing, with `--debug` flag, then wait for attachment:
  
  >> ```bash
  >> mpirun -np 2 python main.py --debug <LIST OF RANKS>
  >> ```

* If you debug cuda, you may want to fix the trace by:

  >> ```bash
  >> CUDA_LAUNCH_BLOCKING=1 mpirun -np 2 python main.py --debug <LIST OF RANKS>
  >> ```

* Before you debug, you may want to check run the error is cuda specific and not cpu

## Systems stuff
After talk with Amit Nadav

* maybe do fork to share pinned memory. (need to be done before cuda)
* ```sudo ./pcm.x``` to check data transfers between cpus. (git clone https://github.com/opcm/pcm.git)
* use node 0 : ```numactl --cpunodebind=0 ``` (sudo apt install numactl) 
  * checking this: either ```lstopo``` or ```lspci -vvv | less```. 
* ```less /proc/$PID/maps``` to check memory mapping
* check allocated sutff with ```strace -f python -v ```
* There was 
  * ```sudo apt install hwloc``` for something.
  * ```modprobe msr``` for something.

## Crazy ideas

* connect all layers to the last one to improve optimization. (theoretical works show this can eliminate bad local minima). (motivation is Competitors can't do it, their partitioning does not support it)

* we can utilize idle/spare time to do gradient smoothing.
  * inspired from [this very bad nips paper which I accidently read](http://papers.nips.cc/paper/9402-theoretical-limits-of-pipeline-parallel-optimization-and-application-to-distributed-deep-learning)

* increace momentum?

## NOTICE
* some ga resulst ran without GA! re running.... (some: thous after thier application was before decoupling step for last partition) I rerun all gap aware just in case...


* I see that with 2 buffers we are actually slower, probobly the time to create 2 buffers not worth it.

## Building (pytorch, ompenmpi)
see conda recipe for cuda aware openmpi.

To see how pytorch is compiled, use 
``` 
torch.__config__.show() 
```
