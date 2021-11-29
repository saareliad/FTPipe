
## Available pipes

Altough our [publication](https://www.usenix.org/system/files/atc21-eliad.pdf) refers mainly to 1-2  pipeline approachs for fine-tuning giant models on commodity hardware (mainly, the Pareto frontiers for the discussed setting), the framework we implemented (quite a while before the publication) supports training all model sizes with, for which, of course, different sweetspots apply.

We implemented many pipeline optimization algorithms to study the tradeoffs of DNN training with asynchronous pipeline-parallelism.

The following pipeline configurations are available:

<!-- ### Stale,  pipelines -->

- `stale`: no staleness mitigation.


- weight prediction (`wp`) : {`msnag`, `aggmsnag`}
   - supported for the {`sgd`,`adam`,`adamw`}` optimizers
   - `msnag` is momentum based weight prediction
   - `aggmsnag` is adopting momentum based wieght prediction to gradient accumulation

- recomputation
   - See Table 1 on [FTPipe paper](https://www.usenix.org/system/files/atc21-eliad.pdf) for the effect on stale pipelines
- no recomputation  (`nr` or `norecomp`)

- weight stashing (`ws`)

- [Gap Aware](https://arxiv.org/pdf/1909.10802.pdf) staleness mitigation (`ga`)
  - for {`sgd`, `adam`, `adamw`} optimizers 
- scheduler aware prediction: making the weight prediction aware of the scheduler.
- gradient aggregation in pipeline (`step_every`)

- combinations of mostly all of the above: {`wp`, `ws`, `ga`}

Note: Weight predicion is often called `msnag` in code.


### Fully-synchronous

- `gpipe`
- DistributedDataParallel (DDP): SSGD
- Sequential (`seq`): naive inter-layer model parallelisem (multi gpu)
- and of course, a single gpu for small models.


Note: Tied weights are handled (decorated) per use-case.