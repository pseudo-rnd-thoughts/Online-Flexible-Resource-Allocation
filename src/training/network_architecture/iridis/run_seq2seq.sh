#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=24:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running Seq2Seq resource weighting agent'
python -m training.network_architecture.seq2seq_network