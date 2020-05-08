#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=48:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running TD3 agent'
python -m training.algorithm.td3