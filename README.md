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

## Bugs

* gpu bcast not working yet. (some deadlock)
  * The problem is that p0 does 'send' bcast to p1 while p1 does 'send' bcast to p0, on the same group. Therefore need more groups.
* We need to manualy `wait()` on async handlers, otherwise memory explodes.

* With `torch.distributed.launch` we may need to manually do ```kill -9 <PIDs>``` in case of errors to kill workers.

* We currently must do `drop_last=True` because we use constant buffer buffers
  (we can write code to fix that, but we prefer this easy fix meanwhile).

* `in_place` operations (like ReLU) at partition border have a potential of destroying our saved activations (?).

* fix batch normalization in `torch.no_grad()`

## TODOs

* Double Buffering (Remember credit to Mark)
* Monkey patch batch normalization: is just a moneky patch.
  * can write spesific class to every case (e.g Batch Norm) to make it more efficient.
* timing (ms per batch, fwd, bwd, etc)
* schedule: find way to eliminate bubbles
* Target_tensor_names:
  * I notice that they (pipedream) pass the target all across the pipeline (waste of energy, inefficient).
  * This can be solved using two data-loaders with correct synchronization.
  * sometimes (in LM) we don't even need to send the "y" as its created from the "x".

* CUDA aware openmpi
  * test to make sure we get the desired speedup [unlike this guy](https://www.pugetsystems.com/labs/hpc/P2P-peer-to-peer-on-NVIDIA-RTX-2080Ti-vs-GTX-1080Ti-GPUs-1331/#test-setup-and-results).

* Support multi node after everything works.
* later change `batch_idx` to `micro_batch_index` to be consistent with the paper.

* Can do gap aware at step 2 even on deeper partitions (effective only once per epoch)
* gap aware with wieght stashing
* "send the layer not the gradient"
* add "scheduler aware prediction".
* can wait in the last layer (extra irecv) to overlap (more) communication with computation.
  * this requires another rcv buffer but in the last partition we don't have mem problems.

* change c code in torch mpigroup to drop the tensor (and its pointers) once completed, so we won't have to handle mem cleaning (this can reduce peak memory memory).

* option in GA to change gradient before sending! (I put this default True).
* look at detach_(), remove uneeded.


## References

* [running with mpi](https://www.open-mpi.org/faq/?category=running)
  * especially see [mpi-env-vars](https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables).

## Debugging

* [debugging mpi python applications with vscode](https://gist.github.com/asroy/ca018117e5dbbf53569b696a8c89204f)
  * debug work only when dataloading is on main thread. (`num_data_workers=0`).
  * run same thing, with `--debug` flag, then wait for attachment:
  
  >> ```bash
  >> mpirun -np 2 python main.py --debug
  >> ```

* If you debug cuda, you may want to fix the trace by:

  >> ```bash
  >> CUDA_LAUNCH_BLOCKING=1 mpirun -np 2 python main.py --debug
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

* connect all layers to the last one to improve optimization. (theoretical works show this can eliminate bad local minima). (Competitors can't do it, their partitioning does not support it)

* we can utilize idle/spare time to do gradient smoothing.
  * inspired from [this stupid nips paper](http://papers.nips.cc/paper/9402-theoretical-limits-of-pipeline-parallel-optimization-and-application-to-distributed-deep-learning)

## NOTICE
* some ga resulst ran without GA! re running.... (some: thous after thier application was before decoupling step for last partition) I rerun all gap aware just in case...