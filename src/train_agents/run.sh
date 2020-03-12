#!/bin bash

#SBATCH --time=04:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
conda activate py37env

python -m  train_agents.training.$file

# To submit
# sbatch -p gtx1080 -v file=multi_agent_train run.sh
# sbatch -p gtx1080 -v file=single_agent_train run.sh