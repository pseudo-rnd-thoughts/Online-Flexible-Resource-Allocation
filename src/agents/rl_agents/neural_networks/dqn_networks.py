"""
Dqn networks using variations on the recurrent network
"""

from __future__ import annotations

import gin.tf
import tensorflow as tf

# Todo add regularises to the networks


@gin.configurable
def create_bidirectional_dqn_network(input_width: int, num_actions: int, lstm_width: int = 32, relu_width: int = 32):
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)
    bidirectional_layer = tf.keras.layers.Bidirectional(lstm_layer)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(bidirectional_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear')(relu_layer)

    return tf.keras.Model(name='Bidirectional LSTM Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_lstm_dqn_network(input_width: int, num_actions: int, lstm_width: int = 32, relu_width: int = 32):
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear')(relu_layer)

    return tf.keras.Model(name='LSTM Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_gru_dqn_network(input_width: int, num_actions: int, gru_width: int = 32, relu_width: int = 32):
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    gru_layer = tf.keras.layers.GRU(gru_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(gru_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear')(relu_layer)

    return tf.keras.Model(name='GRU Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_rnn_dqn_network(input_width: int, num_actions: int, rnn_width: int = 32, relu_width: int = 32):
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    rnn_layer = tf.keras.layers.RNN(rnn_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(rnn_layer)
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear')(relu_layer)

    return tf.keras.Model(name='RNN Dqn', inputs=input_layer, outputs=q_layer)


@gin.configurable
def create_lstm_dueling_dqn_network(input_width: int, num_actions: int,
                                    lstm_width: int = 32, relu_width: int = 32, combiner: str = 'avg'):
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    value = tf.keras.layers.Dense(1, activation='linear')(relu_layer)
    advantage = tf.keras.layers.Dense(num_actions, activation='linear')(relu_layer)
    if combiner == 'avg':
        dueling_q_layer = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    elif combiner == 'mean':
        dueling_q_layer = value + (advantage - tf.reduce_max(advantage, axis=1, keepdims=True))
    else:
        raise Exception(f'Unknown combiner function ({combiner})')

    return tf.keras.Model(name='LSTM Dueling Dqn', inputs=input_layer, outputs=dueling_q_layer)


@gin.configurable
def create_lstm_categorical_dqn_network(input_width: int, num_actions: int,
                                        lstm_width: int = 32, relu_width: int = 32, num_atoms: int = 51):
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    distribution_layer = [tf.keras.layers.Dense(num_atoms, activation='linear')(relu_layer)
                          for _ in range(num_actions)]

    return tf.keras.Model(name='LSTM Categorical Dqn', inputs=input_layer, outputs=distribution_layer)
