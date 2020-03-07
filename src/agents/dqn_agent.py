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

        minibatch = rnd.sample(self.replay_buffer, 2)
        gradients = []
        losses = []
        for trajectory in minibatch:
            observation, action, reward, next_observation = trajectory
            assert 0 <= action < self.num_outputs

            with tf.GradientTape() as tape:
                tape.watch(network_variables)

                target = np.array(self.network_model(observation))
                if next_observation is None:
                    target[0][action] = reward
                else:
                    target[0][action] = reward + np.max(self.network_target(next_observation))

                loss = tf.square(target - self.network_model(observation))
                gradients.append(tape.gradient(loss, network_variables))
                losses.append(tf.reduce_max(loss))

        total_gradients = [sum(grad[var] for grad in gradients) for var in range(len(gradients[0]))]
        self.optimiser.apply_gradients(zip(total_gradients, network_variables))

        log.debug(f"Losses: [{', '.join(str(loss) for loss in losses)}]")

    def update_target(self):
        self.network_target.set_weights(self.network_model.get_weights())
