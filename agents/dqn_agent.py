"""Implementation of a deep q-learning network agent"""

from __future__ import annotations

import random as rnd
from collections import deque

import numpy as np
import tensorflow as tf

import core.log as log


class DqnAgent:
    """
    Dqn Agent that can be allocated to multiple servers
    """

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
        self.network_target.set_weights(self.network_model.get_weights())

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
            for pos, trajectory in enumerate(minibatch):
                observation, action, reward, next_observation = trajectory
                target_values[pos] = self.network_model(observation)

                if next_observation:
                    max_next_value = np.max(self.network_target(next_observation))
                    target_values[pos][action] = reward + self.discount_factor * max_next_value - target_values[pos][action]
                else:
                    target_values[pos][action] = reward - target_values[pos][action]

            error = tf.reduce_mean(0.5 * tf.square(target_values))

        log.warning(f'Training {self.name} agent with error {error}')
        network_gradients = tape.gradient(error, network_variables)
        self.optimiser.apply_gradients(zip(network_gradients, network_variables))
