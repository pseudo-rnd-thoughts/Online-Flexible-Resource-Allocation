"""Tests in agent implementation is valid"""

from __future__ import annotations

from typing import List

import tensorflow as tf

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, TaskPricingDdqnAgent, TaskPricingDuelingDqnAgent, \
    ResourceWeightingDqnAgent, ResourceWeightingDdqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dueling_dqn_network, create_lstm_dqn_network, \
    create_bidirectional_dqn_network, create_gru_dqn_network, create_rnn_dqn_network, \
    create_lstm_categorical_dqn_network
from agents.rl_agents.rl_agent import TaskPricingRLAgent
from env.server import Server
from env.task import Task


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
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10), **dqn_tp_arguments),
        TaskPricingDdqnAgent(0, create_lstm_dqn_network(9, 10), **dqn_tp_arguments),
        TaskPricingDuelingDqnAgent(0, create_lstm_dueling_dqn_network(9, 10), **dqn_tp_arguments),
    ]
    for agent in tp_agents:
        assert_args(agent, dqn_tp_arguments)

    rw_agents = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(10, 10), **dqn_rw_arguments),
        ResourceWeightingDdqnAgent(0, create_lstm_dqn_network(10, 10), **dqn_rw_arguments),
        ResourceWeightingDuelingDqnAgent(0, create_lstm_dueling_dqn_network(10, 10), **dqn_rw_arguments),
    ]
    for agent in rw_agents:
        assert_args(agent, dqn_rw_arguments)


def test_agents_build():
    print()
    # All of the Networks
    bidirectional_lstm = create_bidirectional_dqn_network(10, 10)
    lstm = create_lstm_dqn_network(10, 10)
    gru = create_gru_dqn_network(10, 10)
    rnn = create_rnn_dqn_network(10, 10)
    dueling_lstm = create_lstm_dueling_dqn_network(10, 10)
    categorical_lstm = create_lstm_categorical_dqn_network(10, 10)
    # Todo add new networks when created
    print(f'{bidirectional_lstm.name}, {lstm.name}, {gru.name}, {rnn.name}, '
          f'{dueling_lstm.name}, {categorical_lstm.name}')

    # All of the task pricing agents
    tp_dqn_agent = TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10))
    tp_ddqn_agent = TaskPricingDdqnAgent(1, create_lstm_dqn_network(9, 10))
    tp_dueling_dqn_agent = TaskPricingDuelingDqnAgent(2, create_lstm_dqn_network(9, 10))

    # All of the resource weight agents
    rw_dqn_agent = ResourceWeightingDqnAgent(0, create_lstm_dqn_network(10, 10))
    rw_ddqn_agent = ResourceWeightingDdqnAgent(1, create_lstm_dqn_network(10, 10))
    rw_dueling_dqn_agent = ResourceWeightingDuelingDqnAgent(2, create_lstm_dqn_network(10, 10))

    # Todo add new agents when created

    print(f'{tp_dqn_agent.name}, {tp_ddqn_agent.name}, {tp_dueling_dqn_agent.name}')
    print(f'{rw_dqn_agent.name}, {rw_ddqn_agent.name}, {rw_dueling_dqn_agent.name}')


def test_agent_saving():
    print()
    tp_dqn_agent = TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10))

    tp_dqn_agent._save('tmp')

    loaded_model = create_lstm_dqn_network(9, 10)
    loaded_model.load_weights('tmp')


def test_agent_training():
    print()
    agents: List[TaskPricingRLAgent] = [
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10), batch_size=1),
        TaskPricingDdqnAgent(1, create_lstm_dqn_network(9, 10), batch_size=1),
        TaskPricingDuelingDqnAgent(2, create_lstm_dueling_dqn_network(9, 10), batch_size=1),
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
