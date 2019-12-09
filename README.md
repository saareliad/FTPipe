# Multi Step NAG Asynchronous Pipeline

## Setup env

```
make env
```

(or run it step by step)


## Run

```
python -m torch.distributed.launch --nnodes 1 --master_port 6005 --nproc_per_node 2 main.py --cpu
```

## Bugs

* gpu bcast not working yet. (some deadlock)
  * I suspect that a possible problem with gpu bcast vs p2p is that we don't have tags.
* Note/Bug: may need to manually do ```kill -9 <PID>``` in case of errors to kill workers.

## Small TODOs

* timing (per batch)
* a schedule like pipedream (eliminate bubbles)

## TODOs

* Target_tensor_names:
  * I notice that they (pipedream) pass the target all across the pipeline (waste of energy, inefficient).
  * This can be solved using two data-loaders with correct synchronization.
  * sometimes (in LM) we don't even need to send the "y" as its created from the "x".

* CUDA aware openmpi
  * test to make sure we get the desired speedup [unlike this guy](https://www.pugetsystems.com/labs/hpc/P2P-peer-to-peer-on-NVIDIA-RTX-2080Ti-vs-GTX-1080Ti-GPUs-1331/#test-setup-and-results).

* Support multi node after everything works.
* later change `batch_idx` to `micro_batch_index` to be consistent with the paper.

## Crazy ideas

* connect all layers to the last one to improve optimization. (theoretical works show this can eliminate bad local minima).

* we can utilize idle/spare time to do gradient smoothing.
  * inspired from [this stupid nips paper](http://papers.nips.cc/paper/9402-theoretical-limits-of-pipeline-parallel-optimization-and-application-to-distributed-deep-learning)
  