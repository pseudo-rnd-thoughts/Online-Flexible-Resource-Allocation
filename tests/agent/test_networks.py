"""
Checks that neural networks can be differentiated using DQN loss function
"""

from __future__ import annotations

import tensorflow as tf

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network, create_bidirectional_dqn_network, \
    create_rnn_dqn_network, create_gru_dqn_network, create_lstm_dueling_dqn_network, create_lstm_categorical_dqn_network
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


def test_networks():
    print()
    # Environment setup
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, TaskStage.LOADING, 50.0, price=1),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, TaskStage.COMPUTING, 75.0, 10.0, price=1),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, TaskStage.COMPUTING, 72.0, 25.0, price=1)
    ]
    server = Server('Test', 220.0, 35.0, 22.0)

    # Assert the environment is valid
    auction_task.assert_valid()
    for task in allocated_tasks:
        task.assert_valid()
    server.assert_valid()

    # List of networks
    pricing_networks = [
        create_bidirectional_dqn_network(9, 3),
        create_lstm_dqn_network(9, 3),
        create_gru_dqn_network(9, 3),
        create_rnn_dqn_network(9, 3),
        create_lstm_dueling_dqn_network(9, 3),
        create_lstm_dueling_dqn_network(9, 3, combiner='max')
    ]
    weighting_networks = [
        create_bidirectional_dqn_network(16, 3),
        create_lstm_dqn_network(16, 3),
        create_gru_dqn_network(16, 3),
        create_rnn_dqn_network(16, 3),
        create_lstm_dueling_dqn_network(16, 3),
        create_lstm_dueling_dqn_network(16, 3, combiner='max')
    ]

    # Network observations
    auction_obs = tf.expand_dims(TaskPricingDqnAgent.network_obs(auction_task, allocated_tasks, server, 0), axis=0)
    resource_allocation_obs = tf.expand_dims(ResourceWeightingDqnAgent.network_obs(allocated_tasks[0], allocated_tasks, server, 0), axis=0)
    print(f'Auction obs: {auction_obs}')
    print(f'Resource allocation obs: {resource_allocation_obs}')

    # Loop over the networks to find the output and output shape is correct
    for pricing_network, weighting_network in zip(pricing_networks, weighting_networks):
        auction_output = pricing_network(auction_obs)
        resource_output = weighting_network(resource_allocation_obs)
        print(f'Network: {pricing_network.name}'
              f'\n\tAuction output: {auction_output} ({auction_output.shape})'
              f'\n\tResource allocation output: {resource_output} ({resource_output.shape})')
        assert auction_output.shape == (1, 3)
        assert resource_output.shape == (1, 3)

    # Check of the categorical dqn networks as it is a special case
    pricing_network = create_lstm_categorical_dqn_network(9, 3, num_atoms=10)
    weighting_network = create_lstm_categorical_dqn_network(16, 3, num_atoms=10)
    auction_output = pricing_network(auction_obs)
    resource_output = weighting_network(resource_allocation_obs)
    print(f'Network: {pricing_network.name}'
          f'\n\tAuction output: {auction_output}'
          f'\n\tResource allocation output: {resource_output}')
    assert all(output.shape == (1, 10) for output in auction_output)
    assert all(output.shape == (1, 10) for output in resource_output)
