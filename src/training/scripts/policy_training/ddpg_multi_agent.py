"""Initial training of the agents using the basic environments"""

from __future__ import annotations

import gin

from agents.rl_agents.agents.ddpg import TaskPricingDdpgAgent, ResourceWeightingDdpgAgent
from agents.rl_agents.neural_networks.ddpg_networks import create_lstm_actor_network, create_lstm_critic_network
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.scripts.train_agents import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    gin.parse_config_file('./training/settings/standard_config.gin')

    folder = 'ddpg_agents'
    writer, datetime = setup_tensorboard('training/results/logs/', folder)

    save_folder = f'{folder}_{datetime}'

    env = OnlineFlexibleResourceAllocationEnv([
        './training/settings/basic.env',
        './training/settings/large_tasks_servers.env',
        './training/settings/limited_resources.env',
        './training/settings/mixture_tasks_servers.env'
    ])
    eval_envs = generate_eval_envs(env, 5, f'./training/settings/eval_envs/policy_training/')

    task_pricing_agents = [
        TaskPricingDdpgAgent(agent_num, create_lstm_actor_network(9), create_lstm_critic_network(9), save_folder=save_folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingDdpgAgent(0, create_lstm_actor_network(16), create_lstm_critic_network(16), save_folder=save_folder)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 500, task_pricing_agents, resource_weighting_agents, 5)

    for agent in task_pricing_agents:
        agent.save()
    for agent in resource_weighting_agents:
        agent.save()
