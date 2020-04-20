#!/bin/bash

# DQN Agents
sbatch run_dqn_multi_agent.sh
sbatch run_ddqn_multi_agent.sh
sbatch run_dueling_multi_agent.sh

# PG Agents
sbatch run_ddpg_multi_agent.sh
sbatch run_td3_multi_agent.sh