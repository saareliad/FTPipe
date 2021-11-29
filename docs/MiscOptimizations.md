
## Misc

### Communication Matrix Embedding
- Communication Matrix Embedding with cuda P2P samples (15% BW improvment for pipeline). Can use this [script](/misc/p2p_bw_mat.sh).

### Binding to nodes
Binding to CPUs which are closer to GPUs can imporve performance.
- use node 0 : `numactl --cpunodebind=0` (requirement: sudo apt install numactl)
  - checking this: either `lstopo` or `lspci -vvv | less`.

### Check your Pytorch build
- To see how pytorch is compiled, use
```
torch.__config__.show()
```
