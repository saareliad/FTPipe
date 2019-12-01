# Multi Step NAG Asynchronous Pipeline

## Run

```
python -m torch.distributed.launch --nnodes 1 --master_port 6005 --nproc_per_node 2 main.py --cpu
```

## Bugs

* gpu bcast not working yet. (some deadlock)
  * I suspect that a possible problem with gpu bcast vs p2p is that we don't have tags.

## Small TODOs

* timing (per batch)
* a schedule like pipedream (eliminate bubbles)

## TODOs

* Target_tensor_names:
  * I notice that they (pipedream) pass the target all across the pipeline (waste of energy, inefficient).
  * This can be solved using two data-loaders with correct synchronization.
  * sometimes (in LM) we don't even need to send the "y" as its created from the "x".

* CUDA aware openmpi
  * build with pytorch from source
  * https://anaconda.org/teju85/ompi-cuda
  * https://discuss.pytorch.org/t/segfault-using-cuda-with-openmpi/11140
  * https://www.open-mpi.org/faq/?category=runcuda
  * https://github.com/pytorch/pytorch#from-source

* Support multi node after everything works.
* later change `batch_idx` to `micro_batch_index` to be consistent with the paper.

## Crazy ideas

* we can utilize idle/spare time to do gradient smoothing.
  * inspired from [this stupid nips paper](http://papers.nips.cc/paper/9402-theoretical-limits-of-pipeline-parallel-optimization-and-application-to-distributed-deep-learning)
  