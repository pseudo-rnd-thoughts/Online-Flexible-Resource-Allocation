"""
Dqn networks using variations on the recurrent network
"""

from __future__ import annotations

import gin.tf
import tensorflow as tf


@gin.configurable
def create_bidirectional_dqn_network(input_width: int, num_actions: int, lstm_width: int = 32, relu_width: int = 32):
    """
    Creates a bidirectional lstm dqn network

    Args:
        input_width: Network input width
        num_actions: Number of actions
        lstm_width: Number of LSTM layer units
        relu_width: Number of RELU layer units

    Returns: tensorflow keras Model

    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)
    bidirectional_layer = tf.keras.layers.Bidirectional(lstm_layer)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(bidirectional_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear',
                                    kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    return tf.keras.Model(name='Bidirectional_LSTM_Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_lstm_dqn_network(input_width: int, num_actions: int, lstm_width: int = 32, relu_width: int = 32):
    """
    Creates a lstm dqn network

    Args:
        input_width: Network input width
        num_actions: Number of actions
        lstm_width: Number of LSTM layer units
        relu_width: Number of RELU layer units

    Returns: tensorflow keras Model

    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear',
                                    kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    return tf.keras.Model(name='LSTM_Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_gru_dqn_network(input_width: int, num_actions: int, gru_width: int = 32, relu_width: int = 32):
    """
    Creates a gru dqn network

    Args:
        input_width: Network input width
        num_actions: Number of actions
        gru_width: Number of GRU layer units
        relu_width: Number of RELU layer units

    Returns: tensorflow keras Model

    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    gru_layer = tf.keras.layers.GRU(gru_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(gru_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear',
                                    kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    return tf.keras.Model(name='GRU_Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_rnn_dqn_network(input_width: int, num_actions: int, rnn_width: int = 32, relu_width: int = 32):
    """
    Creates a rnn dqn network

    Args:
        input_width: Network input width
        num_actions: Number of actions
        rnn_width: Number of RNN layer units
        relu_width: Number of RELU layer units

    Returns: tensorflow keras Model

    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    rnn_layer = tf.keras.layers.SimpleRNN(rnn_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(rnn_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear',
                                    kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    return tf.keras.Model(name='RNN_Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_lstm_dueling_dqn_network(input_width: int, num_actions: int,
                                    lstm_width: int = 32, relu_width: int = 32, combiner: str = 'avg'):
    """
    Creates a lstm dqn dueling network

    Args:
        input_width: Network input width
        num_actions: Number of actions
        lstm_width: Number of LSTM layer units
        relu_width: Number of RELU layer units
        combiner: Ways of combining the value and advantage layers

    Returns: tensorflow keras Model

    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l1())(lstm_layer)
    value = tf.keras.layers.Dense(1, activation='linear')(relu_layer)
    advantage = tf.keras.layers.Dense(num_actions, activation='linear')(relu_layer)
    if combiner == 'avg':
        dueling_q_layer = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    elif combiner == 'max':
        dueling_q_layer = value + (advantage - tf.reduce_max(advantage, axis=1, keepdims=True))
    else:
        raise Exception(f'Unknown combiner function ({combiner})')

    return tf.keras.Model(name='LSTM_Dueling_Dqn', inputs=input_layer, outputs=dueling_q_layer)


@gin.configurable
def create_lstm_categorical_dqn_network(input_width: int, num_actions: int,
                                        lstm_width: int = 32, relu_width: int = 32, num_atoms: int = 51):
    """
    Creates a lstm categorical dqn network

    Args:
        input_width: Network input width
        num_actions: Number of actions
        lstm_width: Number of LSTM layer units
        relu_width: Number of RELU layer units
        num_atoms: Num of atoms in the output

    Returns: tensorflow keras Model

    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    distribution_layer = tf.keras.layers.Dense(num_atoms * num_actions, activation='softmax',
                                               kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)
    reshape_layer = tf.keras.layers.Reshape((num_actions, num_atoms))(distribution_layer)

    return tf.keras.Model(name='LSTM_Categorical_Dqn', inputs=input_layer, outputs=reshape_layer)
