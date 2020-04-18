"""
Tests that agent implementations; constructor attributes, agent saving, task pricing / resource allocation training
"""

from __future__ import annotations

import tensorflow as tf

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, TaskPricingDdqnAgent, TaskPricingDuelingDqnAgent, \
    ResourceWeightingDqnAgent, ResourceWeightingDdqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dueling_dqn_network, create_lstm_dqn_network

# TODO add comments
from env.environment import OnlineFlexibleResourceAllocationEnv


def test_build_agent():
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
    reinforcement_learning_arguments = {
        'batch_size': 16, 'optimiser': tf.keras.optimizers.Adadelta(),
        'error_loss_fn': tf.compat.v1.losses.mean_squared_error, 'initial_training_replay_size': 1000,
        'update_frequency': 2, 'replay_buffer_length': 20000, 'save_frequency': 12500, 'save_folder': 'test'
    }
    dqn_arguments = {'target_update_tau': 1.0, 'target_update_frequency': 2500, 'discount_factor': 0.9}
    pricing_arguments = {'failed_auction_reward': -100, 'failed_multiplier': -100}
    weighting_arguments = {'other_task_discount': 0.2, 'success_reward': 1, 'failed_reward': -2,
                           'reward_multiplier': 2.0, 'ignore_empty_next_obs': False}

    dqn_pricing_arguments = {**reinforcement_learning_arguments, **dqn_arguments, **pricing_arguments}
    dqn_weighting_arguments = {**reinforcement_learning_arguments, **dqn_arguments, **weighting_arguments}

    pricing_network = create_lstm_dqn_network(9, 10)
    pricing_agents = [
        TaskPricingDqnAgent(0, pricing_network, **dqn_pricing_arguments),
        TaskPricingDdqnAgent(1, pricing_network, **dqn_pricing_arguments),
        TaskPricingDuelingDqnAgent(2, pricing_network, **dqn_pricing_arguments),
    ]
    for agent in pricing_agents:
        assert_args(agent, dqn_pricing_arguments)

    weighting_network = create_lstm_dqn_network(16, 10)
    weighting_agents = [
        ResourceWeightingDqnAgent(0, weighting_network, **dqn_weighting_arguments),
        ResourceWeightingDdqnAgent(1, weighting_network, **dqn_weighting_arguments),
        ResourceWeightingDuelingDqnAgent(2, weighting_network, **dqn_weighting_arguments),
    ]
    for agent in weighting_agents:
        assert_args(agent, dqn_weighting_arguments)


def test_saving_agent():
    print()
    dqn_agent = TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10))

    dqn_agent._save('agent/checkpoints/dqn_agent')

    loaded_model = create_lstm_dqn_network(9, 10)
    loaded_model.load_weights('agent/checkpoints/dqn_agent')

    print(f'Model: {dqn_agent.model_network.variables}, Loaded: {loaded_model.variables}')
    assert all(tf.reduce_all(weights == load_weights)
               for weights, load_weights in zip(dqn_agent.model_network.variables, loaded_model.variables))


def test_agent_actions():
    print()
    pricing_agents = [
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 5)),
        TaskPricingDdqnAgent(1, create_lstm_dqn_network(9, 5)),
        TaskPricingDuelingDqnAgent(2, create_lstm_dueling_dqn_network(9, 5))
    ]
    weighting_agents = [
        ResourceWeightingDqnAgent(3, create_lstm_dqn_network(16, 5)),
        ResourceWeightingDdqnAgent(4, create_lstm_dqn_network(16, 5)),
        ResourceWeightingDuelingDqnAgent(5, create_lstm_dueling_dqn_network(16, 5))
    ]

    env, state = OnlineFlexibleResourceAllocationEnv.load_env('agent/settings/actions.env')
    actions = {
        server: pricing_agents[pos].bid(state.auction_task, tasks, server, state.time_step)
        for pos, (server, tasks) in enumerate(state.server_tasks.items())
    }
    print(f'Actions: {{{", ".join([f"{server.name}: {action}" for server, action in actions.items()])}}}')

    state, rewards, done, _ = env.step(actions)

    actions = {
        server: weighting_agents[pos].weight(tasks, server, state.time_step)
        for pos, (server, tasks) in enumerate(state.server_tasks.items())
    }
    print(f'Actions: {{{", ".join([f"{server.name}: {list(task_action.values())}" for server, task_action in actions.items()])}}}')
    assert any(0 < action for server, task_actions in actions.items() for task, action in task_actions.items())

    state, rewards, done, _ = env.step(actions)
