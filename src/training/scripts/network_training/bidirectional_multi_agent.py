"""Training of agents using the bidirectional lstm network"""

from __future__ import annotations

import gin

from agents.rl_agents.agents.dqn import ResourceWeightingDqnAgent, TaskPricingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_bidirectional_dqn_network
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.scripts.train_agents import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    # Parse configs
    gin.parse_config_file('./training/settings/standard_config.gin')

    # Setup tensorboard
    folder = 'bidirectional_lstm_agents'
    writer = setup_tensorboard('training/results/logs/', folder)

    # Load the environment
    env = OnlineFlexibleResourceAllocationEnv('./training/settings/basic.env')
    eval_envs = generate_eval_envs(env, 5, f'./training/settings/eval_envs/{folder}/')

    # Setup the pricing agents
    task_pricing_agents = [
        TaskPricingDqnAgent(agent_num, create_bidirectional_dqn_network(9, 10), save_folder=folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingDqnAgent(0, create_bidirectional_dqn_network(16, 10), save_folder=folder)
    ]

    # Train the agents
    with writer.as_default():
        run_training(env, eval_envs, 450, task_pricing_agents, resource_weighting_agents, 5)

    for agent in task_pricing_agents:
        agent._save()
    for agent in resource_weighting_agents:
        agent._save()
