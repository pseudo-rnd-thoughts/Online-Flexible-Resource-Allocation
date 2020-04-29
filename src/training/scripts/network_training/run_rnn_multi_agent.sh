#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=16:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running RNN Network DQN multi-agent'
python -m training.scripts.network_training.rnn_multi_agent
