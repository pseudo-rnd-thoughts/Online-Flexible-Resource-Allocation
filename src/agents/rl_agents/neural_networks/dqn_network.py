"""
DQN Networks
"""

import tensorflow as tf


def dqn_lstm_network(input_width: int, num_outputs: int,
                     lstm_width: int = 40, relu_width: int = 20) -> tf.keras.Sequential:
    """
    DQN LSTM networks
    Args:
        input_width: The input width
        num_outputs: The output width
        lstm_width: The LSTM width
        relu_width: The Relu layer width

    Returns: The network

    """
    network = tf.keras.Sequential(
        tf.keras.layers.LSTM(lstm_width, input_shape=[None, input_width]),
        tf.keras.layers.ReLU(relu_width),
        tf.keras.layers.Dense(num_outputs)
    )

    return network


def dqn_bidirectional_lstm_network(input_width: int, num_outputs: int,
                                   lstm_width: int = 40, relu_width: int = 20) -> tf.keras.Sequential:
    """
    DQN Bidirectional LSTM network
    Args:
        input_width: The input width
        num_outputs: The output width
        lstm_width: The LSTM width
        relu_width: The Relu layer width

    Returns: The network

    """
    network = tf.keras.Sequential(
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_width, input_shape=[None, input_width])),
        tf.keras.layers.ReLU(relu_width),
        tf.keras.layers.Dense(num_outputs)
    )

    return network
