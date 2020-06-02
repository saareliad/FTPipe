#!/bin/bash

###
# py_sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py_sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py_sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py_sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NODE=rishon3
NUM_CORES=16
NUM_GPUS=8
JOB_NAME="Pytorch_Gpipe_CTRL"
MAIL_USER="alonde-jager@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=base

sbatch \
	-w $NODE \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o '%x_%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
python $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

