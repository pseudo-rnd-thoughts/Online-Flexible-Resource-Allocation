"""
Implementation of the categorical DQN agent for the task pricing and resource weighting agents
"""

from abc import ABC
from typing import List, Dict

import random as rnd

import gin
import tensorflow as tf
from agents.rl_agents.agents.dqn import DqnAgent
from agents.rl_agents.rl_agents import TaskPricingRLAgent, ResourceWeightingRLAgent, ReinforcementLearningAgent
from env.server import Server
from env.task import Task


@gin.configurable
class CategoricalDqnAgent(DqnAgent, ABC):

    def __init__(self, network: tf.keras.Model, max_value: float = -20.0, min_value: float = 25.0, num_atoms: int = 21,
                 **kwargs):
        DqnAgent.__init__(self, network, **kwargs)

        self.v_min = min_value
        self.v_max = max_value
        self.num_atoms = num_atoms
        self.delta_z = (max_value - min_value) / num_atoms
        self.support = tf.range(min_value, max_value, self.delta_z, dtype=tf.float32)

    def _train(self, states: tf.Tensor, actions: tf.Tensor, next_states: tf.Tensor, rewards: tf.Tensor,
               dones: tf.Tensor) -> float:
        rewards = tf.expand_dims(rewards, axis=-1)
        dones = tf.expand_dims(dones, axis=-1)
        actions = tf.cast(actions, tf.int32)

        network_variables = self.model_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(network_variables)

            # q_logits contains the Q-value logits for all actions.
            q_logits = self.model_network(states)
            reshaped_actions = tf.stack([tf.range(self.batch_size), actions], axis=-1)
            chosen_action_logits = tf.gather_nd(q_logits, reshaped_actions)

            # Next q_logits
            next_target_logits = self.target_network(next_states)
            next_target_probs = tf.nn.softmax(next_target_logits)
            next_target_q_values = tf.reduce_sum(self.support * next_target_probs, axis=-1)
            next_actions = tf.math.argmax(next_target_q_values)
            next_action_indexes = tf.stack([tf.range(self.batch_size), next_actions], axis=-1)
            next_q_distribution = tf.gather_nd(next_target_logits, next_action_indexes)

            # Project the sample Bellman update \hat{T}Z_{\theta} onto the original
            # support of Z_{\theta} (see Figure 1 in paper).
            tiled_support = tf.ones((self.batch_size, self.num_atoms)) * self.support

            target_support = rewards + self.discount_factor * tiled_support * dones

            # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
            clipped_support = tf.expand_dims(tf.clip_by_value(target_support, self.v_min, self.v_max), axis=1)
            tiled_support = tf.tile([clipped_support], [1, 1, self.num_atoms, 1])
            reshaped_target_support = tf.reshape(tf.ones([self.batch_size, 1]) * self.support,
                                                 [self.batch_size, self.num_atoms, 1])
            # numerator = `|clipped_support - z_i|` in Eq7.
            numerator = tf.abs(tiled_support - reshaped_target_support)
            quotient = 1 - (numerator / self.delta_z)
            clipped_quotient = tf.clip_by_value(quotient, 0, 1)

            # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
            inner_prod = clipped_quotient * tf.expand_dims(next_q_distribution, axis=1)
            projection = tf.reduce_sum(inner_prod, 3)[0]

            # Target distribution
            target_distribution = tf.stop_gradient(projection)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(target_distribution, chosen_action_logits))

        grads = tape.gradient(loss, network_variables)
        self.optimizer.apply_gradients(zip(grads, network_variables))

        # Check if to update the target, if so update each variable at a time using the target update tau variable
        if self.total_updates % self.training_freq == 0:
            ReinforcementLearningAgent._update_target_network(self.model_network, self.target_network,
                                                              self.target_update_tau)

        return loss


class TaskPricingCategoricalDqnAgent(CategoricalDqnAgent, TaskPricingRLAgent):

    def __init__(self, agent_num: int, network: tf.keras.Model, **kwargs):
        CategoricalDqnAgent.__init__(self, network, **kwargs)
        TaskPricingRLAgent.__init__(self, f'Task pricing C51 agent {agent_num}', **kwargs)

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int,
                    training: bool = False):
        if training:
            self._update_epsilon()
            if rnd.random() < self.epsilon:
                return float(rnd.randint(0, self.num_actions - 1))

        observation = tf.expand_dims(self._network_obs(auction_task, allocated_tasks, server, time_step), axis=0)
        q_values = self.model_network(observation)
        action = tf.math.argmax(q_values, axis=1, output_type=tf.int32)
        return action


class ResourceWeightingCategoricalDqnAgent(CategoricalDqnAgent, ResourceWeightingRLAgent):

    def __init__(self, agent_num: int, network: tf.keras.Model, **kwargs):
        CategoricalDqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingRLAgent.__init__(self, f'Resource weighting C51 agent {agent_num}', **kwargs)

    def _get_actions(self, tasks: List[Task], server: Server, time_step: int,
                     training: bool = False) -> Dict[Task, float]:
        if training:
            self._update_epsilon()

            actions = {}
            for task in tasks:
                if rnd.random() < self.epsilon:
                    actions[task] = float(rnd.randint(0, self.num_actions - 1))
                else:
                    observation = tf.expand_dims(self._network_obs(task, tasks, server, time_step), axis=0)
                    q_values = self.model_network(observation) * self.support
                    actions[task] = float(tf.math.argmax(q_values, axis=1, output_type=tf.int32))
            return actions
        else:
            observations = tf.convert_to_tensor([self._network_obs(task, tasks, server, time_step) for task in tasks],
                                                dtype='float32')
            q_values = self.model_network(observations) * self.support
            actions = tf.math.argmax(q_values, axis=1, output_type=tf.int32)
            return {task: float(action) for task, action in zip(tasks, actions)}
