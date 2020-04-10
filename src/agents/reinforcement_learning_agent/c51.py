import tensorflow as tf

from agents.reinforcement_learning_agent.dqn import DqnAgent


class C51Agent(DqnAgent):

    def __init__(self, network: tf.keras.Model, name: str, min_value: float, max_value: float, num_atoms: int = 51,
                 **kwargs):
        DqnAgent.__init__(self, network, name=name, loss=tf.keras.losses.CategoricalCrossentropy(), **kwargs)

        self.z = tf.linspace(min_value, max_value, num_atoms)

    def _loss(self, states, actions, next_states, rewards, dones):
        q_logits = self.model_network(states)
        action_q_logits = self._action_values(q_logits, actions)

        next_q_logits = self.target_network(next_states)
        next_actions = tf.math.argmax(tf.reduce_sum(self.z * next_q_logits, axis=2), axis=1)
        next_actions_q_logits = self._action_values(next_q_logits, next_actions)
