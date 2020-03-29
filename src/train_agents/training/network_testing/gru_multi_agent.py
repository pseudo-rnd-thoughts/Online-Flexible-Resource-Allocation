"""Training of the agent using the gru network"""

from __future__ import annotations

import gin

from agents.rl_agents.dqn import ResourceWeightingDqnAgent, TaskPricingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnGruNetwork
from env.environment import OnlineFlexibleResourceAllocationEnv
from train_agents.core import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    gin.parse_config_file('./train_agents/training/standard_config.gin')

    folder = 'gru_agents'
    writer = setup_tensorboard(folder)

    env = OnlineFlexibleResourceAllocationEnv.make('./train_agents/env_settings/basic_env.json')
    eval_envs = generate_eval_envs(env, 5, f'./train_agents/eval_envs/{folder}/')

    task_pricing_agents = [
        TaskPricingDqnAgent(agent_num, DqnGruNetwork(9, 10), save_folder=folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingDqnAgent(agent_num, DqnGruNetwork(10, 10), save_folder=folder)
        for agent_num in range(3)
    ]

    print('TP Agents: [' + ', '.join(agent.name for agent in task_pricing_agents) + ']')
    print('RW Agents: [' + ', '.join(agent.name for agent in resource_weighting_agents) + ']')

    with writer.as_default():
        run_training(env, eval_envs, 150, task_pricing_agents, resource_weighting_agents, 5)

    for agent in task_pricing_agents:
        agent.save()
    for agent in resource_weighting_agents:
        agent.save()

    print('TP Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}' for agent in task_pricing_agents) + '}')
    print('RW Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}' for agent in resource_weighting_agents) + '}')
