import random as rnd
from collections import deque
from copy import copy

import tensorflow as tf
from tf_agents.utils import common


class DqnAgent:

    def __init__(self, name: str, network: tf.keras.Model, replay_buffer_length: int = 10000, num_actions: int = 2,
                 batch_size: int = 32, optimiser=tf.keras.optimizers.Adam(), discount_factor: float = 0.9):
        self.name = name

        self.model_network = network
        self.target_network = copy(network)
        self.target_network.set_weights(self.model_network.get_weights().copy())

        self.replay_buffer = deque(maxlen=replay_buffer_length)

        self.loss_func = tf.keras.losses.Huber()
        self.optimiser = optimiser

        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.total_steps = 0
        self.training_steps = 0

        self.num_actions = num_actions

        self.train = common.function(self.train)

    def observation(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def train(self):
        states, actions, next_states, rewards, dones = zip(*rnd.sample(self.replay_buffer, self.batch_size))
        states, actions, next_states, rewards, dones = tf.convert_to_tensor(states), tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(next_states), tf.convert_to_tensor(rewards), tf.convert_to_tensor(dones)
        # states = tf.keras.preprocessing.sequence.pad_sequences(states, padding='post', dtype='float32')
        # next_states = tf.keras.preprocessing.sequence.pad_sequences(next_states, padding='post', dtype='float32')

        network_variable = self.model_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(network_variable)

            loss = self._loss(states, actions, next_states, rewards, dones)
        grads = tape.gradient(loss, network_variable)
        self.optimiser.apply_gradients(zip(grads, network_variable))

        return loss

    def _loss(self, states, actions, next_states, rewards, dones):
        q_values = self._compute_q_values(states, actions)
        next_q_values = self._compute_next_q_values(next_states)

        td_targets = rewards + self.discount_factor * next_q_values * dones

        loss = tf.reduce_mean(input_tensor=self.loss_func(td_targets, q_values)) + \
            tf.reduce_mean(self.model_network.losses)

        return loss

    def _compute_q_values(self, states, actions):
        q_values = self.model_network(states)

        return self._action_values(q_values, actions)

    def _compute_next_q_values(self, next_states):
        next_q_values = self.target_network(next_states)
        next_actions = tf.math.argmax(next_q_values, axis=0)

        return self._action_values(next_q_values, next_actions)

    def _action_values(self, q_values, actions):
        actions = tf.concat([tf.expand_dims(tf.range(self.num_actions), -1)] + [actions], -1)
        return tf.gather_nd(q_values, actions)


class DdqnAgent(DqnAgent):

    def __init__(self, name: str, network: tf.keras.Model, **kwargs):
        super().__init__(name, network, **kwargs)

    def _compute_next_q_values(self, next_states):
        next_q_values = self.target_network(next_states)
        next_actions = tf.math.argmax(self.model_network(next_states), axis=0)

        return self._action_values(next_q_values, next_actions)
