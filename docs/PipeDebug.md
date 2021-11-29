## Debugging

- [debugging mpi python applications with vscode](https://gist.github.com/asroy/ca018117e5dbbf53569b696a8c89204f)

  - debug work only when dataloading is on main thread. (`num_data_workers=0`).
  - run same thing, with `--debug` flag, then wait for attachment:

  > > ```bash
  > > mpirun -np 2 python main.py --debug <LIST OF RANKS>
  > > ```

- If you debug cuda, you may want to fix the trace by:

  > > ```bash
  > > CUDA_LAUNCH_BLOCKING=1 mpirun -np 2 python main.py --debug <LIST OF RANKS>
  > > ```

- Before you debug, you may want to check run the error is cuda specific and not cpu

