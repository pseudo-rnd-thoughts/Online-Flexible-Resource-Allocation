import tensorflow as tf

from agents.dqn import DqnAgent


class C51Agent(DqnAgent):

    def __init__(self, name: str, network: tf.keras.Model, min_value: float, max_value: float, **kwargs):
        super().__init__(name, network, optimiser=tf.keras.losses.CategoricalCrossentropy(), **kwargs)

    def _compute_next_q_values(self, next_states):
        return super()._compute_next_q_values(next_states)

