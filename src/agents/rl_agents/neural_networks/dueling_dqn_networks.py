"""Dueling DQN Networks"""

from __future__ import annotations

import gin.tf
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DuelingDqnLstmNetwork(Network):
    """
    Dueling DQN LSTM network
    """

    def __init__(self, input_width: int, num_actions: int, lstm_width: int = 40, relu_width: int = 20):
        super().__init__('Dueling Lstm', input_width, num_actions)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width, input_shape=(None, input_width))
        self.relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')

        self.advantage_layer = tf.keras.layers.Dense(num_actions, activation='linear')
        self.value_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation through the neural network

        Args:
            inputs: numpy ndarray input observation for the network
            training: Ignored
            mask: Ignored

        Returns: Single dimensional array action output

        """
        lstm = self.lstm_layer(inputs)
        relu = self.relu_layer(lstm)

        advantage = self.advantage_layer(relu)
        value = self.value_layer(relu)

        action_q_value = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return action_q_value
