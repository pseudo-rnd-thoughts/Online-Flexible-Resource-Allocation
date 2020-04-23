"""Initial training of the agents using the basic environments"""

from __future__ import annotations

import gin

from agents.rl_agents.agents.dqn import TaskPricingDuelingDqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dueling_dqn_network
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.scripts.train_agents import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    gin.parse_config_file('./training/settings/standard_config.gin')

    folder = 'dueling_multi_agents'
    writer = setup_tensorboard('training/results/logs/', folder)

    env = OnlineFlexibleResourceAllocationEnv('./training/settings/basic.env')
    eval_envs = generate_eval_envs(env, 5, f'./training/settings/eval_envs/{folder}/')

    task_pricing_agents = [
        TaskPricingDuelingDqnAgent(agent_num, create_lstm_dueling_dqn_network(9, 10), save_folder=folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingDuelingDqnAgent(0, create_lstm_dueling_dqn_network(16, 10), save_folder=folder)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 300, task_pricing_agents, resource_weighting_agents, 5)

    for agent in task_pricing_agents:
        agent._save()
    for agent in resource_weighting_agents:
        agent._save()
