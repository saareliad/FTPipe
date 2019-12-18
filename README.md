# Multi Step NAG Asynchronous Pipeline

## Setup env

```bash
make env
```

(or run it step by step)

## Run

### MPI

cuda aware openmpi:

```bash
mpirun -np 2 python main.py
```

on the rishon sever:

```bash
salloc -n2 --gres=gpu:2 mpirun python main.py
```

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
* Monkey patch batch normalization
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

## References

* [running with mpi](https://www.open-mpi.org/faq/?category=running)
  * especially see [mpi-env-vars](https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables).

## Debugging

* [debugging mpi python applications with vscode](https://gist.github.com/asroy/ca018117e5dbbf53569b696a8c89204f)
  * TODO: debug for MPI stopped working somehow, need to fix. Can debug gloo on cpu if lazy.

  * run same thing, with `--debug` flag, then wait for attachment:
  
  >> ```bash
  >> mpirun -np 2 python main.py --debug
  >> ```

* If you debug cuda, you may want to fix the trace by:

  >> ```bash
  >> CUDA_LAUNCH_BLOCKING=1 mpirun -np 2 python main.py --debug
  >> ```

* Before you debug, you may want to check run the error is cuda specific and not cpu

## Crazy ideas

* connect all layers to the last one to improve optimization. (theoretical works show this can eliminate bad local minima). (Competitors can't do it, their partitioning does not support it)

* we can utilize idle/spare time to do gradient smoothing.
  * inspired from [this stupid nips paper](http://papers.nips.cc/paper/9402-theoretical-limits-of-pipeline-parallel-optimization-and-application-to-distributed-deep-learning)
  