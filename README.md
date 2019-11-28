# TODOs

* Target_tensor_names:
  * I notice that they (pipedream) pass the target all across the pipeline (waste of energy, inefficient).
  * This can be solved using two data-loaders with correct synchronization.
  * sometimes (in LM) we don't even need to send the "y" as its created from the "x".

* CUDA aware openmpi
  * build with pytorch from source

