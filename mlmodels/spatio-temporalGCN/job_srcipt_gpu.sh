#!/bin/bash
#SBATCH --constraint gpu
#SBATCH --account m1248
#SBATCH --qos regular
#SBATCH --job-name hep-gpu
#SBATCH --time 240
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=bmohammed@lbl.gov
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --dependency singleton
#SBATCH --exclusive
#SBATCH --gpus-per-task 1
#SBATCH --output %x-%j.out
#SBATCH --error %x-%j.err

module load tensorflow/2.5.0-gpu
module load pytorch/1.7.1-gpu
module load python/3.9-anaconda-2020.11
export PYTHONPATH=$PYTHONPATH:/global/common/cori_cle7/software/tensorflow/2.5.0-gpu/lib/python3.9/site-packages/
export PYTHONPATH=$PYTHONPATH:/global/common/cori_cle7/software/python/3.9-anaconda-2021.11/lib/python3.9/site-packages
export PYTHONPATH=$PYTHONPATH:/global/common/cori_cle7/software/python/3.9-anaconda-2021.11/lib/python3.9/site-packages/numpy/core/__init__.py

srun python3 main.py 