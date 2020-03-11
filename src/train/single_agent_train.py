"""Initial training of the agents using the basic environments"""

from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnBidirectionalLstmNetwork
from env.environment import OnlineFlexibleResourceAllocationEnv
from train.core import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    setup_tensorboard('basic_training_logs')

    env = OnlineFlexibleResourceAllocationEnv.make('../env_settings/basic_env.json')
    eval_envs = generate_eval_envs(env, 5, 'basic_training_eval_envs')

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

    run_training(env, eval_envs, 5000, task_pricing_dqn_agents, resource_weighting_dqn_agents, 5)

    print(f'TP Total Obs: {task_pricing_dqn_agents[0].name}: {task_pricing_dqn_agents[0].total_obs}')
    print(f'RW Total Obs: {resource_weighting_dqn_agents[0].name}: {resource_weighting_dqn_agents[0].total_obs}')
