# Partitioning
 Readme WIP.
 
 ## Pitfalls
 sometimes ops are traced with `training=True`, so replace, e.g:
 
 ```bash
 sed "s/training=True/training=self.training/" op_* | grep training= 
```