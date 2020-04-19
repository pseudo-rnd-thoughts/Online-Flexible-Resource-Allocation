"""
Ddpg networks using variations on the recurrent network
"""

from __future__ import annotations

import gin.tf
import tensorflow as tf

# Todo add regularises to the networks


@gin.configurable
def create_lstm_actor_network(input_width: int, lstm_width: int = 32, relu_width: int = 32):
    """
    Creates the LSTM actor network

    Args:
        input_width: The input width
        lstm_width: LSTM layer width
        relu_width: RELU layer width

    Returns: tf.keras.Model actor
    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_width)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    action = tf.keras.layers.Dense(1, activation='relu')(relu_layer)

    return tf.keras.Model(name='LSTM_Actor', inputs=input_layer, outputs=action)


@gin.configurable
def create_lstm_critic_network(input_width: int, lstm_width: int = 32, relu_width: int = 32):
    """
    Creates the LSTM critic network

    Args:
        input_width: The input width
        lstm_width: LSTM layer width
        relu_width: RELU layer width

    Returns: tf.keras.Model critic
    """
    input_layer = tf.keras.layers.Input(shape=(None, input_width))
    action_input_layer = tf.keras.layers.Input(shape=(1,))
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    concat = tf.keras.layers.Concatenate([lstm_layer, action_input_layer])
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(concat)
    q_values = tf.keras.layers.Dense(1, activation='linear')(relu_layer)

    return tf.keras.Model(name='LSTM_Critic', inputs=[input_layer, action_input_layer], outputs=q_values)
