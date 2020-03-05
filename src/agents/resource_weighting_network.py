"""Resource weighting networks"""

from __future__ import annotations

import tensorflow as tf


class ResourceWeightingNetwork(tf.keras.Model):
    """Resource weighting network using three layer network - LSTM tasks, ReLU layer, linear layer"""

    input_width = 16

    def __init__(self, lstm_connections: int = 10, relu_connections: int = 20, num_outputs: int = 25):
        super().__init__()

        self.task_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_connections),
                                                        input_shape=[None, self.input_width])
        self.relu_layer = tf.keras.layers.Dense(relu_connections, activation='relu')
        self.q_layer = tf.keras.layers.Dense(num_outputs, activation='linear')

    def call(self, inputs, training=None, mask=None):
        """
        Propagates the forward-call of the network
        :param inputs: The inputs
        :param training: Unused variable
        :param mask: Unused variable
        :return: The output of the network
        """
        task_output = self.task_layer(inputs)
        relu_output = self.relu_layer(task_output)
        return self.q_layer(relu_output)
