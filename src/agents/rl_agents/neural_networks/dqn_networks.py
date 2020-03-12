"""
DQN Networks
"""

import tensorflow.keras.layers as tf

from agents.rl_agents.neural_networks.network import Network


class DqnBidirectionalLstmNetwork(Network):
    """
    DQN Bidirectional Lstm Network
    """

    def __init__(self, input_width: int, action_width: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Bidirectional Lstm', input_width, action_width)

        self.bidirectional_lstm = tf.Bidirectional(tf.LSTM(lstm_width, input_shape=[None, input_width]))
        self.relu = tf.ReLU(relu_width)
        self.q_value_layer = tf.Dense(action_width)

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation through the neural network

        Args:
            inputs: numpy ndarray input observation for the network
            training: Ignored
            mask: Ignored

        Returns: Single dimensional array action output

        """
        return self.q_value_layer(self.relu(self.bidirectional_lstm(inputs)))


class DqnLstmNetwork(Network):
    """
    DQN Lstm Network
    """

    def __init__(self, input_width: int, action_width: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Lstm', input_width, action_width)

        self.lstm = tf.LSTM(lstm_width, input_shape=(None, input_width))
        self.relu = tf.ReLU(relu_width)
        self.q_value_layer = tf.Dense(action_width)

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation through the neural network

        Args:
            inputs: numpy ndarray input observation for the network
            training: Ignored
            mask: Ignored

        Returns: Single dimensional array action output

        """
        return self.q_value_layer(self.relu(self.lstm(inputs)))


class DqnGruNetwork(Network):
    """
    DQN Gru Network
    """

    def __init__(self, input_width: int, action_width: int, lstm_width: int = 40, relu_width: int = 20):
        Network.__init__(self, 'Gru', input_width, action_width)

        self.gru = tf.GRU(lstm_width, input_shape=[None, input_width])
        self.relu = tf.ReLU(relu_width)
        self.q_value_layer = tf.Dense(action_width)

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation through the neural network

        Args:
            inputs: numpy ndarray input observation for the network
            training: Ignored
            mask: Ignored

        Returns: Single dimensional array action output

        """
        return self.q_value_layer(self.relu(self.gru(inputs)))
