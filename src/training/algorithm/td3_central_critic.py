"""
Training of Td3 agents using a central critic for the task pricing agents
"""

from __future__ import annotations

import gin

from agents.rl_agents.agents.ddpg import ResourceWeightingTD3Agent, \
    TaskPricingTD3Agent
from agents.rl_agents.neural_networks.ddpg_networks import create_lstm_actor_network, create_lstm_critic_network
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.train_agents import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    folder = 'td3_central_critic_agent'
    writer, datetime = setup_tensorboard('training/results/logs/', folder)

    save_folder = f'{folder}_{datetime}'

    env = OnlineFlexibleResourceAllocationEnv([
        './training/settings/basic.env',
        './training/settings/large_tasks_servers.env',
        './training/settings/limited_resources.env',
        './training/settings/mixture_tasks_servers.env'
    ])
    eval_envs = generate_eval_envs(env, 20, f'./training/settings/eval_envs/algo/')

    central_critic = create_lstm_critic_network(9)
    central_twin_critic = create_lstm_critic_network(9)
    task_pricing_agents = [
        TaskPricingTD3Agent(agent_num, create_lstm_actor_network(9), central_critic, central_twin_critic,
                            save_folder=save_folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingTD3Agent(0, create_lstm_actor_network(16), create_lstm_critic_network(16),
                                  create_lstm_critic_network(16), save_folder=save_folder)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 600, task_pricing_agents, resource_weighting_agents, 10)

    for agent in task_pricing_agents:
        agent.save()
    for agent in resource_weighting_agents:
        agent.save()
