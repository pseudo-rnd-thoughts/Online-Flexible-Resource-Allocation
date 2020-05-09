#!/bin/bash

#SBATCH --time=20:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

module load cplex/12.8

echo 'Running Fixed env analysis'
python -m analysis.fixed_heuristics.analyse_fixed
