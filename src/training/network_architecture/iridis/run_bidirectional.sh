#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=24:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running Bidirectional network architecture'
python -m training.network_architecture.bidirectional_network
