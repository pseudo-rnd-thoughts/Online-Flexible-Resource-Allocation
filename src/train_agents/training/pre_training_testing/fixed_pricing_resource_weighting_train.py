
"""Initial training of the agents using the basic environments"""

from __future__ import annotations

import gin

from agents.heuristic_agents.fixed_task_pricing_agent import FixedTaskPricingAgent
from agents.rl_agents.dqn import ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork
from env.environment import OnlineFlexibleResourceAllocationEnv
from train_agents.core import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    gin.parse_config_file('../standard_config.gin')

    folder = 'fixed_pricing_resource_weighting'
    writer = setup_tensorboard(folder)

    env = OnlineFlexibleResourceAllocationEnv.make('./train_agents/env_settings/basic_env.json')
    eval_envs = generate_eval_envs(env, 5, f'./train_agents/eval_envs/{folder}/')

    fixed_task_pricing_agents = [
        FixedTaskPricingAgent(agent_num, 3) for agent_num in range(3)
    ]
    resource_weighting_dqn_agents = [
        ResourceWeightingDqnAgent(agent_num, DqnLstmNetwork(10, 10), save_folder=folder)
        for agent_num in range(3)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 150, fixed_task_pricing_agents, resource_weighting_dqn_agents, 5)

    print('TP Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}'
                                        for agent in fixed_task_pricing_agents) + '}')
    print('RW Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}'
                                        for agent in resource_weighting_dqn_agents) + '}')
