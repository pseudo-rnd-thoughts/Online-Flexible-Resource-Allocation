
import tensorflow as tf


class DuelingDqnLstmNetwork(tf.keras.Sequential):
    """
    Dueling DQN LSTM network
    """

    def __init__(self, input_width: int, num_actions: int, lstm_width: int, relu_width: int):
        super().__init__()

        self.lstm_layer = tf.keras.layers.LSTM(lstm_width, input_shape=[None, input_width])
        self.relu_layer = tf.keras.layers.ReLU(relu_width)

        self.advantage_layer = tf.keras.layers.ReLU(num_actions)
        self.value_layer = tf.keras.layers.ReLU(1)

    def call(self, inputs, training=None, mask=None):
        """
        Todo
        Args:
            inputs:
            training:
            mask:

        Returns:

        """
        lstm = self.lstm_layer(inputs)
        relu = self.relu_layer(lstm)

        advantage = self.advantage_layer(relu)
        value = self.value_layer(relu)

        action_q_value = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return action_q_value


def dueling_dqn_lstm_network(input_width: int, num_actions: int,
                             lstm_width: int = 40, relu_width: int = 20) -> tf.keras.Sequential:
    """
    Dueling DQN LSTM networks
    Args:
        input_width: The input width
        num_actions: The output width
        lstm_width: The LSTM width
        relu_width: The Relu layer width

    Returns: The network

    """

    return DuelingDqnLstmNetwork(input_width, num_actions, lstm_width, relu_width)
