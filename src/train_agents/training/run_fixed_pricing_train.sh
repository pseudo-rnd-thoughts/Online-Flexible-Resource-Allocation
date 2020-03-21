#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=12:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running fixed pricing train'
python -m  train_agents.training.fixed_pricing_resource_weighting_train

# sbatch run_fixed_pricing_train.sh