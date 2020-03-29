"""Checks that neural networks can be differentiated using DQN loss function"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from agents.rl_agents.ddpg import TaskPricingDdpgAgent
from agents.rl_agents.dqn import TaskPricingDqnAgent
from agents.rl_agents.neural_networks.ddpg_networks import DdpgActorLstmNetwork, DdpgCriticLstmNetwork
from agents.rl_agents.neural_networks.distributional_networks import DistributionalLstmNetwork
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork, DqnBidirectionalLstmNetwork, DqnGruNetwork
from agents.rl_agents.neural_networks.dueling_dqn_networks import DuelingDqnLstmNetwork
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


# noinspection DuplicatedCode
def test_network_gradients():
    print()
    network = DqnLstmNetwork(9, 4)

    server = Server('Test', 220.0, 35.0, 22.0)
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0,
             compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
    ]
    optimiser = tf.keras.optimizers.RMSprop(lr=0.001)
    loss_func = tf.keras.losses.MeanSquaredError()

    obs = TaskPricingDqnAgent.network_obs(auction_task, allocated_tasks, server, 0)
    network(obs)

    network_variables = network.trainable_variables
    with tf.GradientTape() as tape:
        obs = TaskPricingDqnAgent.network_obs(auction_task, allocated_tasks, server, 0)

        action_q_values = network(obs)
        print(f'Action Q Values: {action_q_values}')

        loss = loss_func(np.array([[0, 0, 0, 0]]), action_q_values)
        print(f'Loss: {loss}')

    network_gradients = tape.gradient(loss, network_variables)
    print(f'Network Gradients: {network_gradients}')
    optimiser.apply_gradients(zip(network_gradients, network_variables))


# noinspection DuplicatedCode
def test_network_obs():
    print()
    network = DqnLstmNetwork(9, 15)
    network.build()

    server = Server('Test', 220.0, 35.0, 22.0)
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0,
             compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
    ]

    agent = TaskPricingDqnAgent(0, network)
    for pos in range(len(allocated_tasks)+1):
        obs = network(agent.network_obs(auction_task, allocated_tasks[:pos], server, 0))
        print(obs)


# noinspection DuplicatedCode
def test_network_input_shape():
    print()

    network = tf.keras.Sequential([
        tf.keras.layers.LSTM(10, input_shape=(None, 9)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='linear')
    ])
    network.build()
    network.summary()

    server = Server('Test', 220.0, 35.0, 22.0)
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0,
             compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
    ]

    for pos in range(len(allocated_tasks) + 1):
        obs = network(TaskPricingDqnAgent.network_obs(auction_task, allocated_tasks[:pos], server, 0))
        print(obs)


def test_network_output_shape():
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0,
             compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
    ]
    server = Server('Test', 220.0, 35.0, 22.0)

    dqn_networks = {
        DqnLstmNetwork(9, 3): (1, 3),
        DqnBidirectionalLstmNetwork(9, 3): (1, 3),
        DqnGruNetwork(9, 3): (1, 3),
        DuelingDqnLstmNetwork(9, 3): (1, 3),
        DistributionalLstmNetwork(9, 3, num_atoms=10): (3, 1, 10),
    }

    dqn_obs = TaskPricingDqnAgent.network_obs(auction_task, allocated_tasks, server, 0)
    for network, output_shape in dqn_networks.items():
        output = network(dqn_obs)
        print(network.name, output, output.shape)
        print()
        assert output.shape == output_shape, f'{output}, {output.shape}'

    actor_network = DdpgActorLstmNetwork(9)
    critic_network = DdpgCriticLstmNetwork(9)
    actor_obs = TaskPricingDdpgAgent.actor_network_obs(auction_task, allocated_tasks, server, 0)
    print('Ddpg actor obs', actor_obs, actor_obs.shape)
    assert all(len(ob) == 9 for ob in actor_obs[0])
    action = actor_network(actor_obs)
    print(actor_network.name, action, action.shape)
    assert action.shape == (1, 1), f'{action}, {action.shape}'

    critic_obs = TaskPricingDdpgAgent.critic_network_obs(auction_task, allocated_tasks, server, 0, action)
    q_value = critic_network(critic_obs)
    print(critic_network.name, q_value, q_value.shape)
    assert q_value.shape == (1, 1), f'{q_value}, {q_value.shape}'
