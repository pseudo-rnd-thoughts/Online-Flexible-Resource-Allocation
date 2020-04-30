#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=16:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running LSTM network DQN multi-agent'
python -m training.network_training.lstm_multi_agent
