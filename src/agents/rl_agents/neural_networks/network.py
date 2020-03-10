"""Abstract neural network"""

from abc import ABC

import tensorflow as tf


class Network(tf.keras.Model, ABC):
    """
    Abstract network
    """

    def __init__(self, name, input_width, max_action_value):
        super().__init__()

        self.network_name = name
        self.input_width = input_width
        self.max_action_value = max_action_value
