# TODOs

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

## Crazy ideas

* we can utilize idle/spare time to do gradient smoothing.
  * inspired from [this stupid nips paper](http://papers.nips.cc/paper/9402-theoretical-limits-of-pipeline-parallel-optimization-and-application-to-distributed-deep-learning)
  