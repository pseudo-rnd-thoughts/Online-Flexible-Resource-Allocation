"""Initial training of the agents using the basic environments"""

import datetime as dt

from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork
from env.environment import OnlineFlexibleResourceAllocationEnv
from train_agents.core import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    current_time = dt.datetime.now().strftime("%m%d_%H%M%S")
    folder = 'multi_agents'
    writer = setup_tensorboard(f'logs/{folder}_{current_time}')

    env = OnlineFlexibleResourceAllocationEnv.make('../env_settings/basic_env.json')
    eval_envs = generate_eval_envs(env, 5, f'../eval_envs/{folder}/')

    task_pricing_dqn_agents = [
        TaskPricingDqnAgent(agent_num, DqnLstmNetwork(agent_num, 9, 10), save_frequency=260, save_folder=folder,
                            replay_buffer_length=50000, training_replay_start_size=20000,
                            target_update_frequency=10000, final_exploration_frame=100000)
        for agent_num in range(1)
    ]
    resource_weighting_dqn_agents = [
        ResourceWeightingDqnAgent(agent_num, DqnLstmNetwork(agent_num, 10, 10), save_frequency=260, save_folder=folder,
                                  replay_buffer_length=50000, training_replay_start_size=20000,
                                  target_update_frequency=10000, final_exploration_frame=100000)
        for agent_num in range(1)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 6, task_pricing_dqn_agents, resource_weighting_dqn_agents, 5)

    print('TP Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}' for agent in task_pricing_dqn_agents) + '}')
    print('RW Total Obs: {' + ', '.join(
        f'{agent.name}: {agent.total_obs}' for agent in resource_weighting_dqn_agents) + '}')
