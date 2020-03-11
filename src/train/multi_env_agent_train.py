"""Initial training of the agents using the basic environments"""

from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnBidirectionalLstmNetwork
from env.environment import OnlineFlexibleResourceAllocationEnv
from train.core import generate_eval_envs, run_training, setup_tensorboard

if __name__ == "__main__":
    setup_tensorboard('multi_agent_logs')

    env = OnlineFlexibleResourceAllocationEnv.make([
        '../env_settings/basic_env.json'
    ])
    eval_envs = generate_eval_envs(env, 10, 'multi_agent_eval_envs')

    task_pricing_dqn_agents = [
        TaskPricingDqnAgent(agent_num, DqnBidirectionalLstmNetwork(agent_num, 9, 10),
                            replay_buffer_length=20000, training_replay_start_size=10000,
                            target_update_frequency=1000)
        for agent_num in range(3)
    ]
    resource_weighting_dqn_agents = [
        ResourceWeightingDqnAgent(agent_num, DqnBidirectionalLstmNetwork(agent_num, 10, 10),
                                  replay_buffer_length=50000, training_replay_start_size=25000,
                                  target_update_frequency=10000)
        for agent_num in range(3)
    ]

    run_training(env, eval_envs, 5000, task_pricing_dqn_agents, resource_weighting_dqn_agents, 5)

    print('TP Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}' for agent in task_pricing_dqn_agents) + '}')
    print('RW Total Obs: {' + ', '.join(f'{agent.name}: {agent.total_obs}' for agent in resource_weighting_dqn_agents) + '}')
