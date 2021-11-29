## Usage

Note: The full readme is still WIP. However there is a partial recipe below for adding new task/model and some [examples](https://github.com/saareliad/FTPipe/tree/master/t5_used_scripts_example) of recent scripts we used to concudct some experiments. Feel free to contact.


0. clone the repository.
1. To run partitioning, prepare a `Task`, which is simply a model and example inputs for it. Place it [here](autopipe/tasks).

2. Choose partitioning and analysis settings, for example:
    ```bash
    #  The folloiwing will partition a wide-resnet model with group normalization to 4 GPUs, using PipeDream's search, assuming image size of 3x32x32 - e.g., CIFAR datasets)
    python -m autopipe.partition vision --crop 32 --no_recomputation -b 256 -p 4 --save_memory_mode --partitioning_method pipedream --model wrn_28x10_c100_dr03_gn
    
    ```
  
    This will create, compile, and autogenerate the partitioned model and automatically place it [here](models/partitioned).
    
    _Note: some hyper-parameters in mpipe partitioning, env and so on are still hardcoded and not available as cmd options._

3. Register the partitioned model to the pipeline runtime. 

    In our experiments, this is done by implementing a `CommonModelHandler`, which handles this logic 
    ([see examples](pipe/models/registery)). Note that some models may require additional settings.

4. Register a new dataset or use an existing one.  

    In our experiments, this is done by implementing a `CommonDatasetHandler`.
    The logic for doing so is [here](pipe/data).
    
    Note that additional logic is added to prevent unnecessary data movements automatically.

5. Define training and staleness mitigation settings. Then, run experiments with desired settings.

    In our experiments, this is done by passing a json configuration ([examples](pipe/configs)).
   An example run:
    ```bash
   python -m pipe.main --config pipe/configs/cv/cifar100/wrn28x10/no_recomputation/stale_nr.json --bs_train_from_cmd --bs_train 16 --step_every_from_cmd --step_every 16 --seed 42 --mode mp
   ```    
    Finally, a json file with results will be created and placed as defined in the config.
