#!/bin bash

#SBATCH --time=04:00:00

cd Online-Flexible-Resource-Allocation/src/training/

module load conda
conda activate py37env

python multi_agent_train.py

# To submit
# sbatch -p gtx1080 run.sh