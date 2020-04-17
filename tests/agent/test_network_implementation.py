"""
Checks that neural networks can be differentiated using DQN loss function
"""

from __future__ import annotations

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network, create_bidirectional_dqn_network, \
    create_rnn_dqn_network, create_gru_dqn_network, create_lstm_dueling_dqn_network, create_lstm_categorical_dqn_network
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage

# TODO Add comments


def test_network_output_shape():
    print()
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0,
             compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
    ]
    server = Server('Test', 220.0, 35.0, 22.0)

    dqn_networks = {
        create_bidirectional_dqn_network(9, 3): (1, 3),
        create_lstm_dqn_network(9, 3): (1, 3),
        create_gru_dqn_network(9, 3): (1, 3),
        create_rnn_dqn_network(9, 3): (1, 3),
        create_lstm_dueling_dqn_network(9, 3): (1, 3),
        create_lstm_dueling_dqn_network(9, 3, combiner='max'): (1, 3),
    }

    dqn_obs = TaskPricingDqnAgent.network_obs(auction_task, allocated_tasks, server, 0)
    for network, output_shape in dqn_networks.items():
        output = network(dqn_obs)
        print(f'Network: {network.name}, Output: {output}, shape: {output.shape}')
        print()
        assert output.shape == output_shape, f'{output}, {output.shape}'

    network = create_lstm_categorical_dqn_network(9, 3, num_atoms=10)
    output = network(dqn_obs)
    print(f'Network: {network.name}, Output: {output}')
    assert len(output) == 3 and all(output_layer.shape == (1, 10) for output_layer in output)
