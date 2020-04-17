"""
Tests that agent implementations; constructor attributes, agent saving, task pricing / resource allocation training
"""

from __future__ import annotations

import tensorflow as tf

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, TaskPricingDdqnAgent, TaskPricingDuelingDqnAgent, \
    ResourceWeightingDqnAgent, ResourceWeightingDdqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dueling_dqn_network, create_lstm_dqn_network

# TODO add comments


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

    dqn_agent._save('agent/checkpoints/dqn_agent')

    loaded_model = create_lstm_dqn_network(9, 10)
    loaded_model.load_weights('agent/checkpoints/dqn_agent')

    print(f'Model: {dqn_agent.model_network.variables}, Loaded: {loaded_model.variables}')
    assert all(tf.reduce_all(weights == load_weights)
               for weights, load_weights in zip(dqn_agent.model_network.variables, loaded_model.variables))
