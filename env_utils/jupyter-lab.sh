#!/bin/bash

###
# jupyter-lab.sh
#
# This script is intended to help you run jupyter lab on servers.
#
# Example usage:
#
# To run on the gateway machine (limited resources, no GPU):
# ./jupyter-lab.sh
#
# To run on a compute node:
# srun -c 2 --gres=gpu:1 --pty jupyter-lab.sh
#

###
# Conda parameters
#
HH=$HOME
test "$(hostname)" == 'ninja1' && HH=/home_local/$USER
test "$(hostname)" == 'ninja2' && HH=/home_local/$USER
test "$(hostname)" == 'ninja4' && HH=/home_local/$USER
test "$(hostname)" == 'rambo1' && HH=/home_local/$USER
test "$(hostname)" == 'rambo2' && HH=/home_local/$USER
test "$(hostname)" == 'rambo3' && HH=/home_local/$USER
test "$(hostname)" == 'rambo4' && HH=/home_local/$USER
test "$(hostname)" == 'rambo5' && HH=/home_local/$USER

CONDA_HOME=$HH/miniconda3

CONDA_ENV=py38

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

jupyter lab --no-browser --ip=$(hostname -I | cut -d' ' -f1) --port-retries=100

