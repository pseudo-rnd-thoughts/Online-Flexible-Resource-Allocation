"""
Tests the linear exploration method used in the DQN and DDPG agent are correct
"""

import tensorflow as tf

from agents.rl_agents.agents.ddpg import TaskPricingDdpgAgent, ResourceWeightingDdpgAgent
from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent, TaskPricingCategoricalDqnAgent, \
    ResourceWeightingCategoricalDqnAgent
from agents.rl_agents.neural_networks.ddpg_networks import create_lstm_actor_network, create_lstm_critic_network
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network, create_lstm_categorical_dqn_network
from env.environment import OnlineFlexibleResourceAllocationEnv


def test_epsilon_policy():
    print()
    # Tests the epsilon policy by getting agent actions that should update the agent epsilon over time

    env, state = OnlineFlexibleResourceAllocationEnv.load_env('agent/settings/actions.env')

    # Number of epsilon steps for the agents
    epsilon_steps = 25

    # Agents that have a custom _get_action function
    pricing_agents = [
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 5), epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1),
        TaskPricingCategoricalDqnAgent(1, create_lstm_categorical_dqn_network(9, 5), epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1),
        TaskPricingDdpgAgent(2, create_lstm_actor_network(9), create_lstm_critic_network(9), epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1)
    ]
    weighting_agents = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 5), epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1),
        ResourceWeightingCategoricalDqnAgent(1, create_lstm_categorical_dqn_network(16, 5), epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1),
        ResourceWeightingDdpgAgent(2, create_lstm_actor_network(16), create_lstm_critic_network(16), epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1)
    ]

    # Generate a tf writer and generate actions that will update the epsilon values for both agents
    writer = tf.summary.create_file_writer(f'agent/tmp/testing_epsilon')
    num_steps = 10
    with writer.as_default():
        for _ in range(num_steps):
            for agent in pricing_agents:
                actions = {
                    server: agent.bid(state.auction_task, tasks, server, state.time_step, training=True)
                    for server, tasks in state.server_tasks.items()
                }

        state, rewards, done, _ = env.step(actions)

        for _ in range(num_steps):
            for agent in weighting_agents:
                actions = {
                    server: agent.weight(tasks, server, state.time_step, training=True)
                    for server, tasks in state.server_tasks.items()
                }

        state, rewards, done, _ = env.step(actions)

    # Check that the resulting total action are valid
    for agent in pricing_agents:
        print(f'Agent: {agent.name}')
        assert agent.total_actions == num_steps * 3

    for agent in weighting_agents:
        print(f'Agent: {agent.name}')
        assert agent.total_actions == num_steps * 3

    # Check that the agent epsilon are correct
    assert pricing_agents[0].final_epsilon == pricing_agents[0].epsilon and pricing_agents[1].final_epsilon == pricing_agents[1].epsilon
    assert weighting_agents[0].final_epsilon == weighting_agents[0].epsilon and weighting_agents[1].final_epsilon == weighting_agents[1].epsilon
    assert pricing_agents[2].final_epsilon_std == pricing_agents[2].epsilon_std
    assert weighting_agents[2].final_epsilon_std == weighting_agents[2].epsilon_std
