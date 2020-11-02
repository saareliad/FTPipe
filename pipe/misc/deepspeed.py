# need to change
# https://github.com/microsoft/DeepSpeed/blob/01726ce2b8ec1adbffae7974b5bfe600962c2043/deepspeed/runtime/engine.py#L545
# to support other optimizers (adagrad)


# doe they support fp32?
# https://github.com/microsoft/DeepSpeed/issues/109


# {
#   "train_batch_size": 8,
#   "gradient_accumulation_steps": 1,
#   "steps_per_print": 1,
#   "zero_optimization": true,
#   "fp32_allreduce": true,
#   "optimizer": {
#     "type": "Adam",
#     "params": {
#       "lr": 0.0001
#     }
#   },
#
#   "fp16": {
#     "enabled": false
#   }
# }
#

