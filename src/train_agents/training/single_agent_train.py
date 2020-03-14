"""Initial training of the agents using the basic environments"""

from __future__ import annotations

from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork
from env.environment import OnlineFlexibleResourceAllocationEnv
from train_agents.core import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    folder = 'single_agents'
    writer = setup_tensorboard(folder)

    env = OnlineFlexibleResourceAllocationEnv.make('./train_agents/env_settings/basic_env.json')
    eval_envs = generate_eval_envs(env, 5, f'./train_agents/eval_envs/{folder}/')

    task_pricing_ddqn_agents = [
        TaskPricingDqnAgent(0, DqnLstmNetwork(0, 9, 10), save_frequency=25000, save_folder=folder,
                            replay_buffer_length=50000, training_replay_start_size=15000,
                            target_update_frequency=10000, final_exploration_frame=100000)
    ]
    resource_weighting_ddqn_agents = [
        ResourceWeightingDqnAgent(0, DqnLstmNetwork(0, 10, 10), save_frequency=25000,
                                  save_folder=folder, replay_buffer_length=50000, training_replay_start_size=15000,
                                  target_update_frequency=10000, final_exploration_frame=100000)
    ]

    print('TP Agents: [' + ', '.join(agent.name for agent in task_pricing_ddqn_agents) + ']')
    print('RW Agents: [' + ', '.join(agent.name for agent in resource_weighting_ddqn_agents) + ']')

    with writer.as_default():
        run_training(env, eval_envs, 150, task_pricing_ddqn_agents, resource_weighting_ddqn_agents, 5)

    print('TP Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}' for agent in task_pricing_ddqn_agents) + '}')
    print('RW Total Obs: {' + ', '.join(
        f'{agent.name}: {agent.total_obs}' for agent in resource_weighting_ddqn_agents) + '}')
