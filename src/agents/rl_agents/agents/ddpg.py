"""
Continuous action space policy gradient functions
"""

from abc import ABC
from typing import Optional

import tensorflow as tf
from tf_agents.agents import DdpgAgent

from agents.rl_agents.rl_agents import ReinforcementLearningAgent


class DeepDeterministicPolicyGradient(ReinforcementLearningAgent, ABC):
    """
    Deep deterministic policy gradient
    """

    def __init__(self, actor_network: tf.keras.Model, critic_network: tf.keras.Model, **kwargs):
        ReinforcementLearningAgent.__init__(self, **kwargs)

        self.model_actor_network = actor_network
        self.target_actor_network = tf.keras.models.clone_model(actor_network)
        self.model_critic_network = critic_network
        self.target_critic_network = tf.keras.models.clone_model(critic_network)

    def _train(self, states, actions, next_states, rewards, dones) -> float:
        critic_network_variables = self.model_critic_network.trainable_variables
        with tf.GradientTape() as critic_tape:
            critic_tape.watch(critic_network_variables)

            # TODO
            critic_loss = 0
        critic_grads = critic_tape.gradient(critic_loss, critic_network_variables)
        self.optimiser.apply_gradients(zip(critic_grads, critic_network_variables))

        actor_network_variables = self.model_actor_network.trainable_variables
        with tf.GradientTape() as actor_tape:
            actor_tape.watch(actions)
            q_values = self._critic_network((states, actions))
            actions = tf.nest.flatten(actions)
            # TODO
            actor_loss = 0

        actor_grads = actor_tape.gradient(actor_loss, actor_network_variables)
        self.optimiser.apply_gradients(zip(actor_grads, actor_network_variables))

        return critic_loss + actor_loss

    def _save(self, custom_location: Optional[str] = None):
        pass
