 #!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --ntasks-per-node=1
#SBATCH -A m1248
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=bmohammed@lbl.gov
module load python/3.8-anaconda-2020.11
module load tensorflow/intel-1.15.0-py37
module load pytorch/1.7.1
export PYTHONPATH=$PYTHONPATH:/global/u2/m/mohammed/.local/cori/3.8-anaconda-2020.11/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/global/common/cori_cle7/software/python/3.8-anaconda-2020.11/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/global/common/cori_cle7/software/python/3.8-anaconda-2020.11/lib/python3.8/site-packages 

srun python3 main.py 1>> myfile.out 2>> myfile.err