#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=24:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running TD3 auction and Dqn weighting agent'
python -m training.algorithm.td3_auction_dqn_weighting