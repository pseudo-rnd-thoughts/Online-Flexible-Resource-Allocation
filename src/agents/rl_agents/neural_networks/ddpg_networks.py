"""
Ddpg networks using variations on the recurrent network
"""

from __future__ import annotations

import tensorflow as tf


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
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    action = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    return tf.keras.Model(name='LSTM_Actor', inputs=input_layer, outputs=action)


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
    concat = tf.keras.layers.concatenate([lstm_layer, action_input_layer])
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(concat)
    q_values = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    return tf.keras.Model(name='LSTM_Critic', inputs=[input_layer, action_input_layer], outputs=q_values)


def create_seq2seq_actor_network(lstm_width: int = 32):
    """
    Create Seq2Seq actor network

    Args:
        lstm_width: Size of the LSTM network output width

    Returns: Seq2Seq Actor network model
    """
    input_layer = tf.keras.layers.Input(shape=(None, 8))
    encoder = tf.keras.layers.LSTM(lstm_width, return_state=True)
    encoder_output, encoder_state_h, encoder_state_c = encoder(input_layer)  # Ignore the encoder_output

    decoder = tf.keras.layers.LSTM(lstm_width, return_sequences=True)
    decoded = decoder(input_layer, initial_state=[encoder_state_h, encoder_state_c])
    actor_layer = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l1())(decoded)
    return tf.keras.Model(name='Seq2Seq_actor', inputs=input_layer, outputs=actor_layer)


def create_seq2seq_critic_network(lstm_width: int = 32, relu_width: int = 32):
    """
    Create Seq2Seq critic network

    Args:
        lstm_width: Size of the LSTM network output width
        relu_width: Size of the RELU network output width

    Returns: Seq2Seq Critic network model
    """
    task_input_layer = tf.keras.layers.Input(shape=(None, 8))
    action_input_layer = tf.keras.layers.Input(shape=(None, 1))

    concat_layer = tf.keras.layers.concatenate([task_input_layer, action_input_layer])
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(concat_layer)
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)
    q_value = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    return tf.keras.Model(name='Seq2Seq_critic', inputs=[task_input_layer, action_input_layer], outputs=q_value)
