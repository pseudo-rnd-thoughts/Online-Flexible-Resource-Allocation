"""Implementation of a deep q-learning network agent"""

import random as rnd
from collections import namedtuple, deque
from typing import Dict

import numpy as np
import tensorflow as tf

from core.server import Server
import core.log as log

Trajectory = namedtuple('trajectory', ('state', 'action', 'reward', 'next_state'))


class DqnAgent:
    """
    Dqn Agent that can be allocated to multiple servers
    """

    # Enables the use of the same agent on multiple servers by saving the last server observation trajectory
    last_server_trajectory: Dict[Server, Trajectory] = {}

    def __init__(self, name: str, neural_network, num_outputs: int, minibatch_size: int = 32,
                 discount_factor: float = 0.8, replay_buffer_length: int = 8048, learning_rate: float = 0.01,
                 epsilon: float = 0.8, epsilon_min: float = 0.01, epsilon_decay: float = 0.99):
        self.name = name
        self.num_outputs: int = num_outputs
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor

        self.replay_buffer_length: int = replay_buffer_length
        self.replay_buffer: deque = deque(maxlen=replay_buffer_length)

        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay

        self.network_model: tf.keras.Model = neural_network(num_outputs)
        self.network_target: tf.keras.Model = neural_network(num_outputs)

        self.optimiser = tf.keras.optimizers.RMSprop(lr=learning_rate)

    def train(self):
        """
        Trains the agent using an experience replay buffer and a target number
        Example:
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        network_variables = self.network_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(network_variables)

            target_values = np.zeros((self.minibatch_size, self.num_outputs))
            minibatch = rnd.sample(self.replay_buffer, self.minibatch_size)
            for pos, (state, action, reward, next_state) in enumerate(minibatch):
                target_values[pos] = self.network_model(state)

                if next_state:
                    max_next_value = np.max(self.network_target(next_state))
                    target_values[pos][action] = reward + self.discount_factor * max_next_value - target_values[pos][
                        action]
                else:
                    target_values[pos][action] = reward - target_values[pos][action]

            error = tf.reduce_mean(0.5 * tf.square(target_values))

        log.info('Training {} agent with error {}'.format(self.name, error))
        network_gradients = tape.gradient(error, network_variables)
        self.optimiser.apply_gradients(zip(network_gradients, network_variables))
