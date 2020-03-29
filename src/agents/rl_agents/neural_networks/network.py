"""Abstract neural network"""

from __future__ import annotations

from abc import ABC

import tensorflow as tf


class Network(tf.keras.Model, ABC):
    """
    Abstract network model used for the agents
    """

    def __init__(self, name, input_width, output_width):
        tf.keras.Model.__init__(self, name=name)

        self.input_width = input_width
        self.output_width = output_width

    def __str__(self) -> str:
        return f'{self.name} Network - input: {self.input_width}, output: {self.output_width}'
