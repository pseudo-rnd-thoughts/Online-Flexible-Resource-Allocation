"""Tests in agent implementation is valid"""

from __future__ import annotations

from copy import copy
from typing import List

import tensorflow as tf

from agents.rl_agents.ddpg import TaskPricingDdpgAgent, ResourceWeightingDdpgAgent
from agents.rl_agents.ddqn import TaskPricingDdqnAgent, ResourceWeightingDdqnAgent
from agents.rl_agents.distributional_dqn import TaskPricingDistributionalDqnAgent, \
    ResourceWeightingDistributionalDqnAgent
from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.dueling_dqn import TaskPricingDuelingDqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.ddpg_networks import DdpgCriticLstmNetwork, DdpgActorLstmNetwork
from agents.rl_agents.neural_networks.distributional_networks import DistributionalDqnLstmNetwork
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork, DqnBidirectionalLstmNetwork, DqnGruNetwork
from agents.rl_agents.neural_networks.dueling_dqn_networks import DuelingDqnLstmNetwork
from agents.rl_agents.rl_agent import TaskPricingRLAgent, AgentState
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


def test_simple_agent():
    print()
    TaskPricingDqnAgent(0, DqnLstmNetwork(9, 10))
    print(DqnLstmNetwork(9, 10).__str__())


def test_network_copy():
    print()
    network = DqnLstmNetwork(9, 10)
    copy_network = copy(network)
    assert id(network.get_weights()) != id(copy_network.get_weights())
    print(id(network.get_weights()), id(copy_network.get_weights()))
    assert network.get_weights() == copy_network.get_weights()


def test_agent_attributes():
    def assert_args(test_agent, args):
        """
        Asserts that the proposed arguments have assigned to the agent

        Args:
            test_agent: The test agent
            args: The argument used on the agent
        """
        for arg_name, arg_value in args.items():
            assert getattr(test_agent, arg_name) == arg_value, \
                f'Attr: {arg_name}, correct value: {arg_value}, actual value: {getattr(test_agent, arg_name)}'

    # Check inheritance arguments
    dqn_arguments = {
        'target_update_frequency': 100, 'initial_exploration': 0.9, 'final_exploration': 0.2,
        'final_exploration_frame': 100, 'exploration_frequency': 1000, 'loss_func': tf.keras.losses.MeanSquaredError(),
        'clip_loss': False
    }
    tp_arguments = {'failed_auction_reward': -100, 'failed_reward_multiplier': -100}
    rw_arguments = {'other_task_reward_discount': 0.2, 'successful_task_reward': 1, 'failed_task_reward': -2,
                    'task_multiplier': 2.0, 'ignore_empty_next_obs': False}

    dqn_tp_arguments = {**dqn_arguments, **tp_arguments}
    dqn_rw_arguments = {**dqn_arguments, **rw_arguments}

    tp_agents = [
        TaskPricingDqnAgent(0, DqnLstmNetwork(9, 10), **dqn_tp_arguments),
        TaskPricingDdqnAgent(0, DqnLstmNetwork(9, 10), **dqn_tp_arguments),
        TaskPricingDuelingDqnAgent(0, DuelingDqnLstmNetwork(9, 10), **dqn_tp_arguments),
        TaskPricingDistributionalDqnAgent(0, DistributionalDqnLstmNetwork(9, 10), **dqn_tp_arguments)
    ]
    for agent in tp_agents:
        assert_args(agent, dqn_tp_arguments)

    rw_agents = [
        ResourceWeightingDqnAgent(0, DqnLstmNetwork(10, 10), **dqn_rw_arguments),
        ResourceWeightingDdqnAgent(0, DqnLstmNetwork(10, 10), **dqn_rw_arguments),
        ResourceWeightingDuelingDqnAgent(0, DuelingDqnLstmNetwork(10, 10), **dqn_rw_arguments),
        ResourceWeightingDistributionalDqnAgent(0, DistributionalDqnLstmNetwork(10, 10), **dqn_rw_arguments)
    ]
    for agent in rw_agents:
        assert_args(agent, dqn_rw_arguments)


def test_agents_build():
    print()
    bidirectional_lstm = DqnBidirectionalLstmNetwork(10, 10)
    lstm = DqnLstmNetwork(10, 10)
    gru = DqnGruNetwork(10, 10)
    dueling_lstm = DuelingDqnLstmNetwork(10, 10)
    print(f'{bidirectional_lstm.name}, {lstm.name}, {gru.name}, {dueling_lstm.name}')

    tp_dqn_agent = TaskPricingDqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))
    tp_ddqn_agent = TaskPricingDdqnAgent(1, DqnBidirectionalLstmNetwork(9, 10))
    tp_dueling_dqn_agent = TaskPricingDuelingDqnAgent(2, DqnBidirectionalLstmNetwork(9, 10))
    tp_distributional_dqn_agent = TaskPricingDistributionalDqnAgent(3, DistributionalDqnLstmNetwork(9, 10))
    tp_ddpg_agent = TaskPricingDdpgAgent(4, DdpgActorLstmNetwork(9), DdpgCriticLstmNetwork(10))

    rw_dqn_agent = ResourceWeightingDqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))
    rw_ddqn_agent = ResourceWeightingDdqnAgent(1, DqnBidirectionalLstmNetwork(10, 10))
    rw_dueling_dqn_agent = ResourceWeightingDuelingDqnAgent(2, DqnBidirectionalLstmNetwork(10, 10))
    rw_distributional_dqn_agent = ResourceWeightingDistributionalDqnAgent(3, DistributionalDqnLstmNetwork(10, 10))
    rw_ddpg_agent = ResourceWeightingDdpgAgent(4, DdpgActorLstmNetwork(10), DdpgCriticLstmNetwork(11))

    print(f'{tp_dqn_agent.name}, {tp_ddqn_agent.name}, {tp_dueling_dqn_agent.name}, '
          f'{tp_distributional_dqn_agent.name}, {tp_ddpg_agent.name}')
    print(f'{rw_dqn_agent.name}, {rw_ddqn_agent.name}, {rw_dueling_dqn_agent.name}, '
          f'{rw_distributional_dqn_agent.name}, {rw_ddpg_agent.name}')


def test_agent_saving():
    print()
    tp_dqn_agent = TaskPricingDqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))

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


def test_agent_training():
    print()
    agents: List[TaskPricingRLAgent] = [
        TaskPricingDqnAgent(0, DqnLstmNetwork(9, 10), batch_size=1),
        TaskPricingDdqnAgent(1, DqnLstmNetwork(9, 10), batch_size=1),
        TaskPricingDuelingDqnAgent(2, DuelingDqnLstmNetwork(9, 10), batch_size=1),
        TaskPricingDistributionalDqnAgent(3, DistributionalDqnLstmNetwork(9, 10), batch_size=1),
        TaskPricingDdpgAgent(4, DdpgActorLstmNetwork(9), DdpgCriticLstmNetwork(10), batch_size=1)
    ]

    auction_task_1 = Task('auction task 1', 69.0, 35.0, 10.0, 0, 12)
    auction_task_2 = Task('auction task 2', 69.0, 35.0, 10.0, 0, 12)
    tasks = [Task('task 1', 69.0, 35.0, 10.0, 0, 12), Task('task 2', 69.0, 35.0, 10.0, 0, 12)]
    server = Server('server', 220.0, 35.0, 22.0)

    agent_state = AgentState(auction_task_1, tasks, server, 0)
    action = 0
    next_agent_state = AgentState(auction_task_2, tasks, server, 0)

    for agent in agents:
        agent.failed_auction_bid(agent_state, action, next_agent_state)

        agent.train()
