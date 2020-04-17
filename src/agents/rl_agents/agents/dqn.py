"""
Implementation of Deep Q Network, Double DQN and Dueling DQN mechanisms for task pricing and resource allocation agents
"""

from __future__ import annotations

import os
from abc import ABC
from typing import List, Union, Optional, Dict

import gin.tf
import numpy as np
import tensorflow as tf

from agents.rl_agents.rl_agents import ReinforcementLearningAgent, ResourceWeightingRLAgent, TaskPricingRLAgent
from env.server import Server
from env.task import Task

"""
Deep Q Network based on Playing Atari with Deep Reinforcement Learning
 (https://arxiv.org/abs/1312.5602)
"""


@gin.configurable
class DqnAgent(ReinforcementLearningAgent, ABC):
    """
    Deep Q Network agent
    """

    def __init__(self, network: tf.keras.Model, target_update_tau: float = 1.0, target_update_frequency: int = 2500,
                 discount_factor: float = 0.9, **kwargs):
        """
        Constructor for the DQN agent

        Args:
            network_input_width: The network input width
            network_num_outputs: The network num of outputs
            build_network: Function to build networks
            target_update_frequency: The target network update frequency
            **kwargs: Additional arguments for the reinforcement learning agent
        """
        ReinforcementLearningAgent.__init__(self, **kwargs)

        # Set the model and target Q networks
        self.model_network = network
        self.target_network = tf.keras.models.clone_model(network)
        self.num_actions = network.output_shape[1]

        # Target update frequency and tau
        self.target_update_frequency = target_update_frequency
        self.target_update_tau = target_update_tau

        # Discount factor
        self.discount_factor = discount_factor

    def _save(self, custom_location: Optional[str] = None):
        # Set the location to save the model
        if custom_location:
            path = f'{os.getcwd()}/{custom_location}'
        else:
            path = f'{os.getcwd()}/train_agents/results/checkpoint/{self.save_folder}/{self.name.replace(" ", "_")}'

        # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the model network weights to the path
        self.model_network.save_weights(path)

    @tf.function
    def _train(self, states: tf.Tensor, actions: tf.Tensor,
               next_states: tf.Tensor, rewards: tf.Tensor, dones: tf. Tensor) -> float:
        # Actions are discrete so cast to int32 from float32
        actions = tf.cast(actions, tf.int32)

        network_variables = self.model_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(network_variables)

            # Calculate the state q values for the actions
            state_q_values = self.model_network(states)
            state_action_indexes = tf.stack([tf.range(self.batch_size), actions], axis=-1)
            states_actions_q_values = tf.gather_nd(state_q_values, state_action_indexes)

            # Calculate the next state q values for the next actions (important as separate function for double dqn)
            next_states_actions_q_values = self._compute_next_q_values(next_states)
            # Calculate the target using the rewards, discount factor, next q values and dones
            target = tf.stop_gradient(rewards + self.discount_factor * next_states_actions_q_values * dones)

            # Calculate the element wise loss
            loss = self.error_loss_fn(target, states_actions_q_values)

        # Backpropagation the loss through the network variables and apply the changes to the network
        gradients = tape.gradient(loss, network_variables)
        self.optimiser.apply_gradients(zip(gradients, network_variables))

        # Check if to update the target, if so update each variable at a time using the target update tau variable
        if self.total_updates % self.update_frequency == 0:
            for model_variable, target_variable in zip(self.model_network.variables, self.target_network.variables):
                if model_variable.trainable and target_variable.trainable:
                    target_variable.assign(self.target_update_tau * model_variable +
                                           (1 - self.target_update_tau) * target_variable)

        return loss

    def _compute_next_q_values(self, next_states):
        next_state_q_values = self.target_network(next_states)
        next_actions = tf.math.argmax(next_state_q_values, axis=1, output_type=tf.int32)
        next_state_action_indexes = tf.stack([tf.range(self.batch_size), next_actions], axis=-1)
        return tf.gather_nd(next_state_q_values, next_state_action_indexes)


@gin.configurable
class TaskPricingDqnAgent(DqnAgent, TaskPricingRLAgent):
    """
    Task Pricing DQN agent
    """

    network_obs_width: int = 9

    def __init__(self, agent_name: Union[int, str], network: tf.keras.Model, **kwargs):
        assert network.input_shape[-1] == self.network_obs_width

        DqnAgent.__init__(self, network, **kwargs)
        TaskPricingRLAgent.__init__(self, f'DQN TP {agent_name}' if type(agent_name) is int else agent_name, **kwargs)

    @staticmethod
    def network_obs(auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        """
        Network observation for the Q network

        Args:
            auction_task: The pricing task
            allocated_tasks: The allocated tasks
            server: The server
            time_step: The time step

        Returns: numpy ndarray with shape (1, len(allocated_tasks) + 1, 9)

        """

        observation = [ReinforcementLearningAgent.normalise_task(auction_task, server, time_step) + [1.0]] + \
                      [ReinforcementLearningAgent.normalise_task(allocated_task, server, time_step) + [0.0]
                       for allocated_task in allocated_tasks]

        return observation

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        observation = self.network_obs(auction_task, allocated_tasks, server, time_step)
        return tf.math.argmax(self.model_network(observation), axis=0, output_type=tf.float32)


@gin.configurable
class ResourceWeightingDqnAgent(DqnAgent, ResourceWeightingRLAgent):
    """
    Resource weighting DQN agent
    """

    resource_obs_width: int = 10

    def __init__(self, agent_name: Union[int, str], network: tf.keras.Model, **kwargs):
        assert network.input_shape[-1] == self.resource_obs_width
        DqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingRLAgent.__init__(self, f'DQN TP {agent_name}' if type(agent_name) is int else agent_name,
                                          **kwargs)

    @staticmethod
    def network_obs(weighting_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        """
        Network observation for the Q network

        Args:
            weighting_task: The weighing task
            allocated_tasks: The allocated tasks
            server: The server
            time_step: The time step

        Returns: numpy ndarray with shape (1, len(allocated_tasks)-1, self.max_action_value)

        """
        assert 1 < len(allocated_tasks)

        task_observation = ReinforcementLearningAgent.normalise_task(weighting_task, server, time_step)
        observation = [
            task_observation + ReinforcementLearningAgent.normalise_task(allocated_task, server, time_step)
            for allocated_task in allocated_tasks if weighting_task != allocated_task
        ]

        return observation

    def _get_actions(self, tasks: List[Task], server: Server, time_step: int) -> Dict[Task, float]:
        observations = [self.network_obs(task, tasks, server, time_step) for task in tasks]
        actions = tf.math.argmax(self.model_network(observations), axis=1, output_type=tf.int32)
        return {task: action for task, action in zip(tasks, actions)}


"""
Double DQN agent implemented based on Deep Reinforcement Learning with Double Q-learning
 (https://arxiv.org/abs/1509.06461)
"""


@gin.configurable
class DdqnAgent(DqnAgent, ABC):
    """
    Implementation of a double deep q network agent
    """

    def __init__(self, network: tf.keras.Model, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)

    def _compute_next_q_values(self, next_states):
        target_q_values = self.target_network(next_states)
        target_actions = tf.math.argmax(target_q_values, axis=1, output_type=tf.int32)
        target_action_indexes = tf.stack([tf.range(self.batch_size), target_actions], axis=-1)

        model_q_values = self.model_network(next_states)
        return tf.gather_nd(model_q_values, target_action_indexes)


@gin.configurable
class TaskPricingDdqnAgent(DdqnAgent, TaskPricingDqnAgent):
    """
    Task pricing double dqn agent
    """

    def __init__(self, agent_num: int, network: tf.keras.Model, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        TaskPricingDqnAgent.__init__(self, f'DDQN TP {agent_num}', network, **kwargs)


@gin.configurable
class ResourceWeightingDdqnAgent(DdqnAgent, ResourceWeightingDqnAgent):
    """
    Resource weighting double dqn agent
    """

    def __init__(self, agent_num: int, network: tf.keras.Model, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingDqnAgent.__init__(self, f'DDQN RW {agent_num}', network, **kwargs)


"""
Dueling DQN agent based on Dueling Network Architectures for Deep Reinforcement Learning
 (https://arxiv.org/abs/1511.06581)
"""


@gin.configurable
class DuelingDQN(DdqnAgent, ABC):
    """
    Implementations of a dueling DQN agent
    """

    def __init__(self, network: tf.keras.Model, double_loss: bool = False, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)

        if double_loss:
            self._compute_next_q_values = DdqnAgent._compute_next_q_values
        else:
            self._compute_next_q_values = DqnAgent._compute_next_q_values


@gin.configurable
class TaskPricingDuelingDqnAgent(DuelingDQN, TaskPricingDqnAgent):
    """
    Task pricing dueling DQN agent
    """

    def __init__(self, agent_num: int, network: tf.keras.Model, **kwargs):
        DuelingDQN.__init__(self, network, **kwargs)
        TaskPricingDqnAgent.__init__(self, f'Dueling DQN TP {agent_num}', network, **kwargs)


@gin.configurable
class ResourceWeightingDuelingDqnAgent(DuelingDQN, ResourceWeightingDqnAgent):
    """
    Resource Weighting Dueling DQN agent
    """

    def __init__(self, agent_num: int, network: tf.keras.Model, **kwargs):
        DuelingDQN.__init__(self, network, **kwargs)
        ResourceWeightingDqnAgent.__init__(self, f'Dueling DQN RW {agent_num}', network, **kwargs)
