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

        self.network_model: tf.keras.Model = neural_network(num_outputs=num_outputs)
        self.network_target: tf.keras.Model = neural_network(num_outputs=num_outputs)
        self.network_target.set_weights(self.network_model.get_weights())

        self.optimiser = tf.keras.optimizers.RMSprop(lr=learning_rate)

        self.greedy_policy: bool = True

    def train(self):
        """
        Trains the agent using an experience replay buffer and a target model
        """
        log.info(f'Training of {self.name} agent, experience replay length: {len(self.replay_buffer)}')
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        log.debug(f'Updated epsilon to {self.epsilon}')

        network_variables = self.network_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(network_variables)

            model_val = np.zeros((self.minibatch_size, self.num_outputs))
            target_val = np.zeros((self.minibatch_size, self.num_outputs))

            minibatch = rnd.sample(self.replay_buffer, self.minibatch_size)
            for pos, trajectory in enumerate(minibatch):
                observation, action, reward, next_observation = trajectory

                model_val[pos] = np.array(self.network_model(observation))
                target_val[pos] = np.array(self.network_model(observation))

                if next_observation is not None:
                    max_next_value = np.max(self.network_target(next_observation))
                    model_val[pos][action] = reward + self.discount_factor * max_next_value
                else:
                    model_val[pos][action] = reward

            error = tf.square(target_val - model_val)
            error = tf.reduce_mean(0.5 * error)

        log.warning(f'Training {self.name} agent with error {error}')
        network_gradients = tape.gradient(error, network_variables)
        log.warning(f'Network gradients are {network_gradients}')
        self.optimiser.apply_gradients(zip(network_gradients, network_variables))
