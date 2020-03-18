"""
Deep Deterministic Policy Gradient networks (actor and critic)
"""

from __future__ import annotations

import gin.tf
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DdpgLstmActor(Network):
    """
    DDPG actor with LSTM as the primary layer
    """

    def __init__(self, input_width, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'DDPG Actor', input_width, 1)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width)
        self.relu_layer = tf.keras.layers.ReLU(relu_width)
        self.action_layer = tf.keras.layers.ReLU(1)  # As the action space is always greater or equal to zero

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


@gin.configurable
class DdpgLstmCritic(Network):
    """
    DDPG Critic with LSTM as the primary layer
    """

    def __init__(self, input_width, max_action_value, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'DDPG Critic', input_width, max_action_value)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width)
        self.relu_layer = tf.keras.layers.ReLU(relu_width)
        self.q_layer = tf.keras.layers.Dense(1)  # The Q Value for the action

    def call(self, inputs, training=None, mask=None):
        """
        Calls the critic neural network to produce a q value for the action

        Args:
            inputs: The input to run the network with
            training: Ignored
            mask: Ignored

        Returns: The q value for the action

        """
        return self.q_layer(self.relu_layer(self.lstm_layer(inputs)))
