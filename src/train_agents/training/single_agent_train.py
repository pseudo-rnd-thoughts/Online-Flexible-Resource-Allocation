"""Initial training of the agents using the basic environments"""

import datetime as dt

from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnBidirectionalLstmNetwork
from env.environment import OnlineFlexibleResourceAllocationEnv
from train_agents.core import generate_eval_envs, run_training, setup_tensorboard


if __name__ == "__main__":
    current_time = dt.datetime.now().strftime("%m%d_%H%M%S")
    folder = 'single_agents'
    writer = setup_tensorboard(f'logs/{folder}_{current_time}')

    env = OnlineFlexibleResourceAllocationEnv.make('../env_settings/basic_env.json')
    eval_envs = generate_eval_envs(env, 5, f'../eval_envs/{folder}/')

    task_pricing_dqn_agents = [
        TaskPricingDqnAgent(0, DqnBidirectionalLstmNetwork(0, 9, 10),
                            replay_buffer_length=20000, training_replay_start_size=10000,
                            target_update_frequency=1000)
    ]
    resource_weighting_dqn_agents = [
        ResourceWeightingDqnAgent(0, DqnBidirectionalLstmNetwork(0, 10, 10),
                                  replay_buffer_length=50000, training_replay_start_size=25000,
                                  target_update_frequency=10000)
    ]

    with writer.as_default():
        run_training(env, eval_envs, 100, task_pricing_dqn_agents, resource_weighting_dqn_agents, 5)

    print('TP Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}' for agent in task_pricing_dqn_agents) + '}')
    print('RW Total Obs: {' + ', '.join(
        f'{agent.name}: {agent.total_obs}' for agent in resource_weighting_dqn_agents) + '}')
