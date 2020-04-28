"""Initial training of the agents using the basic environments"""

from __future__ import annotations

import gin

from agents.rl_agents.agents.dqn import TaskPricingCategoricalDqnAgent, ResourceWeightingCategoricalDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_categorical_dqn_network
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.scripts.train_agents import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    gin.parse_config_file('./training/settings/standard_config.gin')

    folder = 'c51_agents'
    writer, datetime = setup_tensorboard('training/results/logs/', folder)

    save_folder = f'{folder}_{datetime}'

    env = OnlineFlexibleResourceAllocationEnv([
        './training/settings/basic.env',
        './training/settings/large_tasks_servers.env',
        './training/settings/limited_resources.env',
        './training/settings/mixture_tasks_servers.env'
    ])
    eval_envs = generate_eval_envs(env, 20, f'./training/settings/eval_envs/policy_training/')

    task_pricing_agents = [
        TaskPricingCategoricalDqnAgent(agent_num, create_lstm_categorical_dqn_network(9, 16), save_folder=save_folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingCategoricalDqnAgent(0, create_lstm_categorical_dqn_network(16, 11), save_folder=save_folder)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 500, task_pricing_agents, resource_weighting_agents, 5)

    for agent in task_pricing_agents:
        agent.save()
    for agent in resource_weighting_agents:
        agent.save()
