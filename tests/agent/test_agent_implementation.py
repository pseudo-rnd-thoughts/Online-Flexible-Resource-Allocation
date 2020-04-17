"""
Tests that agent implementations; constructor attributes, agent saving, task pricing / resource allocation training
"""

from __future__ import annotations

from typing import List

import tensorflow as tf

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, TaskPricingDdqnAgent, TaskPricingDuelingDqnAgent, \
    ResourceWeightingDqnAgent, ResourceWeightingDdqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dueling_dqn_network, create_lstm_dqn_network
from agents.rl_agents.rl_agents import TaskPricingRLAgent, TaskPricingState, ResourceWeightingRLAgent, \
    ResourceAllocationState
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.task_stage import TaskStage


# TODO add comments


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
    rl_arguments = {'batch_size': 16, 'optimiser': tf.keras.optimizers.Adadelta(),
                    'error_loss_fn': tf.compat.v1.losses.mean_squared_error, 'initial_training_replay_size': 1000,
                    'update_frequency': 2, 'replay_buffer_length': 20000, 'save_frequency': 12500,
                    'save_folder': 'test'}
    dqn_arguments = {'target_update_tau': 1.0, 'target_update_frequency': 2500, 'discount_factor': 0.9}
    tp_arguments = {'failed_auction_reward': -100, 'failed_multiplier': -100}
    rw_arguments = {'other_task_discount': 0.2, 'success_reward': 1, 'failed_reward': -2,
                    'reward_multiplier': 2.0, 'ignore_empty_next_obs': False}

    dqn_tp_arguments = {**rl_arguments, **dqn_arguments, **tp_arguments}
    dqn_rw_arguments = {**rl_arguments, **dqn_arguments, **rw_arguments}

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


def test_saving_agent():
    print()
    dqn_agent = TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10))

    dqn_agent._save('tmp/dqn_agent')

    loaded_model = create_lstm_dqn_network(9, 10)
    loaded_model.load_weights('agent/tmp/dqn_agent')

    assert all(weights == load_weights
               for weights, load_weights in zip(dqn_agent.model_network.get_weights(), loaded_model.get_weights())), \
        f'Model: {dqn_agent.model_network.get_weights()}, Loaded: {loaded_model.get_weights()}'


def test_task_price_training():
    print()
    # List of agents
    agents: List[TaskPricingRLAgent] = [
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10), batch_size=2),
        TaskPricingDdqnAgent(1, create_lstm_dqn_network(9, 10), batch_size=2),
        TaskPricingDuelingDqnAgent(2, create_lstm_dueling_dqn_network(9, 10), batch_size=2),
    ]

    # Load the environment
    env, state = OnlineFlexibleResourceAllocationEnv.load_env('agent/settings/auction.env')

    # Servers
    server_1, server_2 = list(state.server_tasks.keys())
    # Actions
    actions = {server_1: 1.0, server_2: 2.0}

    # Environment step
    next_state, reward, done, info = env.step(actions)

    # Server states
    server_1_state = TaskPricingState(state.auction_task, state.server_tasks[server_1], server_1, state.time_step)
    server_2_state = TaskPricingState(state.auction_task, state.server_tasks[server_2], server_2, state.time_step)

    # Next servers
    next_server_1 = next(next_server for next_server in next_state.server_tasks.keys() if next_server.name == server_1.name)
    next_server_2 = next(next_server for next_server in next_state.server_tasks.keys() if next_server.name == server_2.name)
    # Next server states
    next_server_1_state = TaskPricingState(next_state.auction_task, next_state.server_tasks[next_server_1],
                                           next_server_1, next_state.time_step)
    next_server_2_state = TaskPricingState(next_state.auction_task, next_state.server_tasks[next_server_2],
                                           next_server_2, next_state.time_step)
    # Finished auction task
    finished_task = next(finished_task for finished_task in next_state.server_tasks[next_server_1] if finished_task == state.auction_task)
    finished_task = finished_task._replace(stage=TaskStage.COMPLETED)

    # Loop over the agents, add the observations and try training
    for agent in agents:
        agent.winning_auction_bid(server_1_state, actions[server_1], finished_task, next_server_1_state)
        agent.failed_auction_bid(server_2_state, actions[server_2], next_server_2_state)

        agent.train()


def test_resource_allocation_training():
    print()
    # List of agents
    agents: List[ResourceWeightingRLAgent] = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(10, 10), batch_size=4),
        ResourceWeightingDdqnAgent(1, create_lstm_dqn_network(10, 10), batch_size=4),
        ResourceWeightingDuelingDqnAgent(2, create_lstm_dueling_dqn_network(10, 10), batch_size=4),
    ]

    # Load the environment
    env, state = OnlineFlexibleResourceAllocationEnv.load_env('agent/settings/resource_allocation.env')

    # Servers and tasks
    server = list(state.server_tasks.keys())[0]
    task_1, task_2, task_3, task_4 = list(state.server_tasks[server])

    # Actions
    actions = {
        server: {
            task_1: 1.0,
            task_2: 3.0,
            task_3: 0.0,
            task_4: 5.0
        }
    }

    # Environment step
    next_state, rewards, done, _ = env.step(actions)

    # Resource state
    resource_state = ResourceAllocationState(state.server_tasks[server], server, state.time_step)
    # Next server and resource state
    next_server = next(next_server for next_server in next_state.server_tasks.keys())
    next_resource_state = ResourceAllocationState(next_state.server_tasks[next_server], next_server, next_state.time_step)

    for agent in agents:
        agent.resource_allocation_obs(resource_state, actions[server], next_resource_state, rewards[next_server])

        agent.train()
