t5_mesh_transformer  \
  --model_dir="model" \
  --gin_file="dataset.gin" \
  --gin_param="utils.run.mesh_shape = 'model:8,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0', 'gpu:1', 'gpu:2', 'gpu:3', 'gpu:4', 'gpu:5', 'gpu:6', 'gpu:7']" \
  --gin_param="MIXTURE_NAME = 'glue_rte_v002'" \
  --gin_param="run.train_steps = 1004000" \
  --gin_param="tokens_per_batch=12800" \
  --gin_param="inputs_length = 320" \
  --gin_param="targets_length = 8" \
  --gin_param="pack_or_pad.pack = False" \
  --gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 1280" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_file="gs://t5-data/pretrained_models/3B/operative_config.gin"



# --gin_param="run.sequence_length = {'inputs': 320, 'targets': 8}"
