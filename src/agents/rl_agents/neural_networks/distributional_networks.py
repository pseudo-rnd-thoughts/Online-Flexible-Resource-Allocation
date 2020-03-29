"""
Distribution DQN network
"""

from __future__ import annotations

import gin.tf
import numpy as np
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DistributionalDqnLstmNetwork(Network):
    """
    Distribution LSTM network
    """

    def __init__(self, input_width: int, num_actions: int, num_atoms: int = 51,
                 lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Distributional DQN', input_width, num_actions)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width)
        self.relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')
        self.action_layers = [tf.keras.layers.Dense(num_atoms, activation='linear') for _ in range(num_actions)]

    def call(self, inputs, training=None, mask=None):
        """
        Calls the actor neural network to produce an action

        Args:
            inputs: The input to run the network with
            training: Ignored
            mask: Ignored

        Returns: The suggested action
        """
        relu_layer_output = self.relu_layer(self.lstm_layer(inputs))
        return np.array([action_layer(relu_layer_output) for action_layer in self.action_layers])
