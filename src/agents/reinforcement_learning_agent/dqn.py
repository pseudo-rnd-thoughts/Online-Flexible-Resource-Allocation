
import tensorflow as tf

from agents.reinforcement_learning_agent.agent import RLAgent


class DqnAgent(RLAgent):

    def __init__(self, network: tf.keras.Model, name: str = 'Dqn Agent',
                 update_target_tau: float = 0.0, target_update_freq: int = 200, **kwargs):
        RLAgent.__init__(self, name=name, **kwargs)
        self.name = name

        network.build()
        self.model_network = network
        self.target_network = tf.keras.models.clone_model(network)

        self.update_target_tau = update_target_tau

        self.target_update_freq = target_update_freq

    @tf.function
    def _train(self, states, actions, next_states, rewards, dones):
        # states = tf.keras.preprocessing.sequence.pad_sequences(states, padding='post', dtype='float32')
        # next_states = tf.keras.preprocessing.sequence.pad_sequences(next_states, padding='post', dtype='float32')

        network_variable = self.model_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(network_variable)

            loss = self._loss(states, actions, next_states, rewards, dones)

        grads = tape.gradient(loss, network_variable)
        self.optimiser.apply_gradients(zip(grads, network_variable))

        return loss

    @tf.function
    def _loss(self, states, actions, next_states, rewards, dones):
        q_values = self._compute_q_values(states, actions)
        target = tf.stop_gradient(rewards + self.discount_factor * self._compute_next_q_values(next_states) * dones)

        loss = tf.reduce_mean(self.loss_func(target, q_values)) + tf.reduce_mean(self.model_network.losses)
        return loss

    @tf.function
    def _compute_q_values(self, states, actions):
        q_values = self.model_network(states)

        return self._action_values(q_values, actions)

    @tf.function
    def _compute_next_q_values(self, next_states):
        next_q_values = self.target_network(next_states)
        next_actions = tf.math.argmax(next_q_values, axis=1)

        return self._action_values(next_q_values, next_actions)

    @tf.function
    def _action_values(self, q_values, actions):
        indexes = tf.cast(tf.range(self.batch_size), tf.int32)
        action_indexes = tf.stack([indexes, actions], axis=-1)

        return tf.gather_nd(q_values, action_indexes)

    def _update_target(self):
        for model_weight, target_weight in zip(self.model_network.variables, self.target_network.variables):
            target_weight.assign((1 - self.update_target_tau) * target_weight + self.update_target_tau * model_weight)


class DdqnAgent(DqnAgent):

    def __init__(self, network: tf.keras.Model, name: str = 'Ddqn Agent', **kwargs):
        super().__init__(network, name, **kwargs)

    def _compute_next_q_values(self, next_states):
        next_q_values = self.target_network(next_states)
        next_actions = tf.math.argmax(self.model_network(next_states), axis=1)

        return self._action_values(next_q_values, next_actions)
