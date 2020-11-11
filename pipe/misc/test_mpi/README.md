## Test MPI run
[adapted from here](https://medium.com/@esaliya/pytorch-distributed-with-mpi-acb84b3ae5fd)
Test if pytorch has openmpi backend

```
mpirun -np 2 python pytorch_distributed.py
```

## Test for multiple machines
on each machine:
```
mpirun --hostfile nodes.txt --map-by node -np 2 python pytorch_distributed.py
```
testing for ninja4 and ninja2 (in this order)

TODO: play with init to make it work (did not work OOTB)
