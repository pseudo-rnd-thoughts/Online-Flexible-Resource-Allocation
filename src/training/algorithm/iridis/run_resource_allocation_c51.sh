#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=20:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running C51 agent'
python -m training.algorithm.resource_allocation_c51
