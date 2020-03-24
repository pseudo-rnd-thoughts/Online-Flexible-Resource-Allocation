"""
Distribution DQN network
"""

from __future__ import annotations

import gin.tf
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DistributionLstmNetwork(Network):

    def __init__(self, input_width: int, max_action_value: int, categories: int = 51,
                 lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Distributional DQN',  input_width, max_action_value)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width)
        self.relu_layer = tf.keras.layers.ReLU(relu_width)
        self.action_layer = tf.keras.layers.Dense(max_action_value * categories)

    def call(self, inputs, training=None, mask=None):
        """
        Calls the actor neural network to produce an action

        Args:
            inputs: The input to run the network with
            training: Ignored
            mask: Ignored

        Returns: The suggested action
        """
        return self.action_layer(self.relu_layer(self.lstm_layer(inputs)))
