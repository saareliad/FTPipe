# Partitioning
 Readme WIP.
 
 Available algorithms under `autopipe.autopipe.model_partitioning`:
 - `mpipe` (mixed-pipe)
 - `pipedream`
 - `metis`
 - `acyclic`
 
 ## Pitfalls
 sometimes ops are traced with `training=True`, so replace, e.g:
 
 ```bash
 sed "s/training=True/training=self.training/" op_* | grep training= 
```