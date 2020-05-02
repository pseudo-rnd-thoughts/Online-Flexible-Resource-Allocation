"""
Tests the linear exploration method used in the DQN agent
"""

# TODO add comments
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
    
    epsilon_steps = 25

    pricing_agents = [
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 5), epsilon_steps=epsilon_steps, epsilon_update_freq=1,
                            epsilon_log_freq=1),
        TaskPricingCategoricalDqnAgent(1, create_lstm_categorical_dqn_network(9, 5), epsilon_steps=epsilon_steps,
                                       epsilon_update_freq=1, epsilon_log_freq=1),
        TaskPricingDdpgAgent(2, create_lstm_actor_network(9), create_lstm_critic_network(9),
                             epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1)
    ]

    weighting_agents = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 5), epsilon_steps=epsilon_steps, epsilon_update_freq=1,
                                  epsilon_log_freq=1),
        ResourceWeightingCategoricalDqnAgent(1, create_lstm_categorical_dqn_network(16, 5), epsilon_steps=epsilon_steps,
                                             epsilon_update_freq=1, epsilon_log_freq=1),
        ResourceWeightingDdpgAgent(2, create_lstm_actor_network(16), create_lstm_critic_network(16),
                                   epsilon_steps=epsilon_steps, epsilon_update_freq=1, epsilon_log_freq=1)
    ]

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

    for agent in pricing_agents:
        print(f'Agent: {agent.name}')
        assert agent.total_actions == num_steps * 3

    for agent in weighting_agents:
        print(f'Agent: {agent.name}')
        assert agent.total_actions == num_steps * 3

    print(f'Initial: {pricing_agents[2].initial_epsilon_std}, '
          f'final: {pricing_agents[2].final_epsilon_std}, '
          f'steps: {pricing_agents[2].epsilon_steps}, '
          f'total actions: {pricing_agents[2].total_actions}')

    assert pricing_agents[2].final_epsilon_std == pricing_agents[2].epsilon_std
    assert weighting_agents[2].final_epsilon_std == weighting_agents[2].epsilon_std


def test_epsilon():
    steps = 10000
    final_epsilon = 0.1
    agent = TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 5), epsilon_steps=steps, initial_epsilon=1,
                                final_epsilon=final_epsilon)
    assert agent.epsilon == 1
    for _ in range(steps+1):
        agent._update_epsilon()

    assert agent.epsilon == final_epsilon
