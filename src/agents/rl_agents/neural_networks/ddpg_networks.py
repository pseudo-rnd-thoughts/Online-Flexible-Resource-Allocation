"""
Deep Deterministic Policy Gradient networks (actor and critic)
"""

from __future__ import annotations

import gin.tf
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DdpgActorLstmNetwork(Network):
    """
    DDPG actor with LSTM as the primary layer
    """

    def __init__(self, input_width: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'DDPG Actor Lstm', input_width, 1)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width, input_shape=(None, input_width))
        self.relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='linear')  # As the action space range is always greater or equal to zero

    def call(self, inputs, training=None, mask=None):
        """
        Calls the actor neural network to produce an action

        Args:
            inputs: The input to run the network with
            training: Ignored
            mask: Ignored

        Returns: The suggested action
        """
        return self.action_layer(self.lstm_layer(inputs))


@gin.configurable
class DdpgCriticLstmNetwork(Network):
    """
    DDPG Critic with LSTM as the primary layer
    """

    def __init__(self, input_width: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'DDPG Critic Lstm', input_width, 1)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width, input_shape=(None, input_width))
        self.relu_layer = tf.keras.layers.ReLU(relu_width)
        self.q_layer = tf.keras.layers.Dense(1, activation='linear')  # The Q Value for the action

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
