"""Tests in agent implementation is valid"""

from __future__ import annotations

from copy import deepcopy, copy

import tensorflow as tf

from agents.rl_agents.ddqn import TaskPricingDdqnAgent, ResourceWeightingDdqnAgent
from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.dueling_dqn import TaskPricingDuelingDqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork, DqnBidirectionalLstmNetwork, DqnGruNetwork
from agents.rl_agents.neural_networks.dueling_dqn_networks import DuelingDqnLstmNetwork
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


def test_simple_agent():
    TaskPricingDqnAgent(0, DqnLstmNetwork(9, 10))


def test_network_copy():
    network = DqnLstmNetwork(9, 10)
    copy_network = copy(network)
    assert id(network.get_weights()) != id(copy_network.get_weights())
    print(id(network.get_weights()), id(copy_network.get_weights()))
    assert network.get_weights() == copy_network.get_weights()


def test_agent_attributes():
    # Check inheritance arguments
    dqn_arguments = {
        'target_update_frequency': 100, 'initial_exploration': 0.9, 'final_exploration': 0.2,
        'final_exploration_frame': 100, 'exploration_frequency': 1000, 'loss_func': tf.keras.losses.MeanSquaredError(),
        'clip_loss': False
    }
    tp_arguments = {'failed_auction_reward': 100, 'failed_reward_multiplier': 100}
    rw_arguments = {'other_task_reward_discount': 0.2, 'successful_task_reward': 1, 'failed_task_reward': -2,
                    'task_multiplier': 2.0, 'ignore_empty_next_obs': False}

    dqn_tp_arguments = {**dqn_arguments, **tp_arguments}
    dqn_tp = TaskPricingDqnAgent(0, DqnBidirectionalLstmNetwork(9, 10), **dqn_tp_arguments)
    for name, value in dqn_tp_arguments.items():
        assert getattr(dqn_tp,
                       name) == value, f'Attr: {name}, correct value: {value}, actual value: {getattr(dqn_tp, name)}'
    assert dqn_tp.name != 'unknown'

    dqn_rw_arguments = {**dqn_arguments, **rw_arguments}
    dqn_rw = ResourceWeightingDqnAgent(0, DqnBidirectionalLstmNetwork(10, 10), **dqn_rw_arguments)
    for name, value in dqn_rw_arguments.items():
        assert getattr(dqn_rw,
                       name) == value, f'Attr: {name}, correct value: {value}, actual value: {getattr(dqn_rw, name)}'
    assert dqn_rw.name != 'unknown'


def test_agents_build():
    print()
    bidirectional_lstm = DqnBidirectionalLstmNetwork(10, 10)
    lstm = DqnLstmNetwork(10, 10)
    gru = DqnGruNetwork(10, 10)
    dueling_lstm = DuelingDqnLstmNetwork(10, 10)
    print(f'{bidirectional_lstm.name}, {lstm.name}, {gru.name}, {dueling_lstm.name}')

    tp_dqn_agent = TaskPricingDqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))
    tp_ddqn_agent = TaskPricingDdqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))
    tp_dueling_dqn_agent = TaskPricingDuelingDqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))

    rw_dqn_agent = ResourceWeightingDqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))
    rw_ddqn_agent = ResourceWeightingDdqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))
    rw_dueling_dqn_agent = ResourceWeightingDuelingDqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))

    assert id(tp_dqn_agent.model_network) != id(tp_dqn_agent.target_network)
    print(f'{tp_dqn_agent.name}, {tp_ddqn_agent.name}, {tp_dueling_dqn_agent.name}')
    print(f'{rw_dqn_agent.name}, {rw_ddqn_agent.name}, {rw_dueling_dqn_agent.name}')


def test_agent_saving():
    print()
    tp_dqn_agent = TaskPricingDqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))

    server = Server('Test', 220.0, 35.0, 22.0)
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0,
             compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
    ]
    tp_dqn_agent.eval_policy = True
    tp_dqn_agent.bid(auction_task, allocated_tasks, server, 0)

    tp_dqn_agent.save()


if __name__ == "__main__":
    test_simple_agent()
