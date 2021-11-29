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


