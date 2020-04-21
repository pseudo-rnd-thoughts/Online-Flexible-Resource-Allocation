"""Initial training of the agents using the basic environments"""

from __future__ import annotations

import gin

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.scripts.train_agents import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    gin.parse_config_file('./training/settings/standard_config.gin')

    folder = 'multi_envs_multi_agents'
    writer = setup_tensorboard('training/results/logs/', folder)

    env = OnlineFlexibleResourceAllocationEnv([
        './train_agents/env_settings/basic.env',
        './train_agents/env_settings/large_tasks_servers.env',
        './train_agents/env_settings/limited_resources.env',
        './train_agents/env_settings/mixture_tasks_servers.env'
    ])
    eval_envs = generate_eval_envs(env, 20, f'./training/settings/eval_envs/{folder}/')

    task_pricing_agents = [
        TaskPricingDqnAgent(agent_num, create_lstm_dqn_network(9, 16), save_folder=folder)
        for agent_num in range(4)
    ]
    resource_weighting_agents = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 10), save_folder=folder)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 600, task_pricing_agents, resource_weighting_agents, 5)

    for agent in task_pricing_agents:
        agent.save()
    for agent in resource_weighting_agents:
        agent.save()
