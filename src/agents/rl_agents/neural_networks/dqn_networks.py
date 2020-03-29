"""Standard DQN Networks"""

from __future__ import annotations

import gin.tf
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DqnBidirectionalLstmNetwork(Network):
    """
    DQN Bidirectional Lstm Network
    """

    def __init__(self, input_width: int, action_width: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Bidirectional Lstm', input_width, action_width)

        lstm_layer = tf.keras.layers.LSTM(lstm_width, input_shape=(None, input_width))
        self.bidirectional_lstm_layer = tf.keras.layers.Bidirectional(lstm_layer)
        self.relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')
        self.q_value_layer = tf.keras.layers.Dense(action_width, activation='linear')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation through the neural network

        Args:
            inputs: numpy ndarray input observation for the network
            training: Ignored
            mask: Ignored

        Returns: Single dimensional array action output

        """
        return self.q_value_layer(self.relu_layer(self.bidirectional_lstm_layer(inputs)))


@gin.configurable
class DqnLstmNetwork(Network):
    """
    DQN Lstm Network
    """

    def __init__(self, input_width: int, num_actions: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Lstm', input_width, num_actions)

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width, input_shape=(None, input_width))
        self.relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')
        self.q_value_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation through the neural network

        Args:
            inputs: numpy ndarray input observation for the network
            training: Ignored
            mask: Ignored

        Returns: Single dimensional array action output

        """
        return self.q_value_layer(self.relu_layer(self.lstm_layer(inputs)))


@gin.configurable
class DqnGruNetwork(Network):
    """
    DQN Gru Network
    """

    def __init__(self, input_width: int, action_width: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Gru', input_width, action_width)

        self.gru_layer = tf.keras.layers.GRU(lstm_width, input_shape=(None, input_width))
        self.relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')
        self.q_value_layer = tf.keras.layers.Dense(action_width, activation='linear')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation through the neural network

        Args:
            inputs: numpy ndarray input observation for the network
            training: Ignored
            mask: Ignored

        Returns: Single dimensional array action output

        """
        return self.q_value_layer(self.relu_layer(self.gru_layer(inputs)))
