
import tensorflow as tf


class DdpgAgent(RLAgent):

    def __init__(self, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 actor_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 critic_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 name: str = 'DDPG Agent', **kwargs):
        RLAgent.__init__(self, name=name, **kwargs)

        actor_network.build()
        self.model_actor_network = actor_network
        self.target_actor_network = tf.keras.models.clone_model(actor_network)
        critic_network.build()
        self.model_critic_network = critic_network
        self.target_critic_network = tf.keras.models.clone_model(critic_network)

        self.actor_optimiser = actor_optimiser
        self.critic_optimiser = critic_optimiser

        self.critic_loss_func = tf.keras.losses.Huber()
        self.actor_loss_func = tf.keras.losses.MeanSquaredError()

    @tf.function
    def _train(self, states, actions, next_states, rewards, dones):
        critic_network_weights = self.model_critic_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(critic_network_weights)
            critic_loss = self._critic_loss(states, actions, next_states, rewards, dones)

        critic_grads = tape.gradient(critic_loss, critic_network_weights)
        self.critic_optimiser.apply_gradients(zip(critic_grads, critic_network_weights))

        actor_network_weights = self.model_actor_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_network_weights)
            actor_loss = self._actor_loss(states)
        actor_grads = tape.gradient(actor_loss, actor_network_weights)
        self.actor_optimiser.apply_gradients(zip(actor_grads, actor_network_weights))

        self._update_target()
        return actor_loss + critic_loss

    def _critic_loss(self, states, actions, next_states, rewards, dones):
        target_actions = self.target_actor_network(next_states)

        target_critic_net_input = (next_states, target_actions)  # Todo combine the state and the actions
        target_q_values = self.target_critic_network(target_critic_net_input)
        td_targets = tf.stop_gradient(rewards + self.discount_factor * target_q_values * dones)

        critic_net_input = (states, actions)  # Todo combine the state and the actions
        q_values = self.model_critic_network(critic_net_input)

        critic_loss = self.critic_loss_func(td_targets, q_values)
        critic_loss = tf.reduce_mean(critic_loss)

        return critic_loss

    def _actor_loss(self, states):
        actions, _ = self.model_actor_network(states)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model_critic_network((states, actions))

        dqdas = tape.gradient([q_values], actions)

        actor_losses = []
        for dqda, action in zip(dqdas, actions):
            loss = self.actor_loss_func(tf.stop_gradient(dqda + action), action)
            loss = tf.reduce_mean(loss)
            actor_losses.append(loss)

        actor_loss = tf.add_n(actor_losses)
        return actor_loss

    def _update_target(self):
        pass
