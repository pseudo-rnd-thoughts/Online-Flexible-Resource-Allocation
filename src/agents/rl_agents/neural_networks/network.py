"""Abstract neural network"""

from __future__ import annotations

from abc import ABC

import tensorflow as tf


class Network(tf.keras.Model, ABC):
    """
    Abstract network model used for the agents
    """

    def __init__(self, name, input_width, max_action_value):
        tf.keras.Model.__init__(self, name=name)

        self.input_width = input_width
        self.max_action_value = max_action_value
