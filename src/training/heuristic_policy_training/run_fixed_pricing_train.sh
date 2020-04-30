#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=12:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running fixed pricing train'
python -m  training.heuristic_policy_training.fixed_pricing_resource_weighting_train
