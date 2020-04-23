"""
Continuous action space policy gradient agents
"""

import os
from abc import ABC
from typing import List, Dict, Union

import tensorflow as tf

from agents.rl_agents.rl_agents import ReinforcementLearningAgent, TaskPricingRLAgent, ResourceWeightingRLAgent
from env.server import Server
from env.task import Task


# Todo agents: D4PG, MADDPG, Seq2Seq DDPG
# Todo work out the value range


class DdpgAgent(ReinforcementLearningAgent, ABC):
    """
    Deep deterministic policy gradient agent
    """

    def __init__(self, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 actor_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
                 critic_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.RMSprop(lr=0.005),
                 initial_epsilon_std: float = 0.8, final_epsilon_std: float = 0.1, epsilon_steps: int = 20000,
                 epsilon_update_frequency: int = 25, min_value: float = -15.0, max_value: float = 15,
                 target_update_tau: float = 1.0, actor_target_update_frequency: int = 3000,
                 critic_target_update_frequency: int = 1500, **kwargs):
        assert actor_network.output_shape[-1] == 1 and critic_network.output_shape[-1] == 1

        ReinforcementLearningAgent.__init__(self, **kwargs)

        # Actor network
        self.model_actor_network = actor_network
        self.target_actor_network: tf.keras.Model = tf.keras.models.clone_model(actor_network)
        self.actor_optimiser = actor_optimiser

        # Critic network
        self.model_critic_network = critic_network
        self.target_critic_network: tf.keras.Model = tf.keras.models.clone_model(critic_network)
        self.critic_optimiser = critic_optimiser

        # Training attributes
        self.min_value = min_value
        self.max_value = max_value
        self.target_update_tau = target_update_tau
        self.actor_target_update_frequency = actor_target_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency

        # Exploration
        self.initial_epsilon_std = initial_epsilon_std
        self.final_epsilon_std = final_epsilon_std
        self.epsilon_grad: float = (initial_epsilon_std - final_epsilon_std) * epsilon_steps
        self.epsilon_std = initial_epsilon_std
        self.epsilon_update_frequency = epsilon_update_frequency

    def _update_epsilon(self):
        self.total_actions += 1
        if self.total_actions % self.epsilon_update_frequency == 0:
            self.epsilon_std = max(self.total_actions / self.epsilon_grad + self.initial_epsilon_std,
                                   self.final_epsilon_std)
            if self.total_actions % 1000 == 0:
                tf.summary.scalar(f'{self.name} agent epsilon std', self.epsilon_std, self.total_actions)

    def _train(self, states, actions, next_states, rewards, dones) -> float:
        # The rewards and dones dims need to be expanded for the td_target to have the same shape as the q values
        rewards, dones = tf.expand_dims(rewards, axis=1), tf.expand_dims(dones, axis=1)

        # Update the critic network
        critic_network_variables = self.model_critic_network.trainable_variables
        with tf.GradientTape() as critic_tape:
            critic_tape.watch(critic_network_variables)

            # Calculate the state and next state q values with the actions and the actor next actions
            state_q_values = self.model_critic_network([states, tf.expand_dims(actions, axis=1)])
            next_actions = self.model_actor_network(next_states)
            next_state_q_values = self.target_critic_network([next_states, next_actions])

            # Calculate the target using the rewards, discount factor, next q values and dones
            td_target = tf.stop_gradient(rewards + self.discount_factor * next_state_q_values * dones)

            # Calculate the element wise loss
            critic_loss = self.error_loss_fn(td_target, state_q_values)
        critic_grads = critic_tape.gradient(critic_loss, critic_network_variables)
        self.critic_optimiser.apply_gradients(zip(critic_grads, critic_network_variables))

        actor_loss = self._actor_loss(states)

        # Check if to update the target, if so update each variable at a time using the target update tau variable
        if self.total_updates % self.actor_target_update_frequency == 0:
            ReinforcementLearningAgent._update_target_network(self.model_actor_network, self.target_actor_network,
                                                              self.target_update_tau)
        if self.total_updates % self.critic_target_update_frequency == 0:
            ReinforcementLearningAgent._update_target_network(self.model_critic_network, self.target_critic_network,
                                                              self.target_update_tau)

        return critic_loss + actor_loss

    def _actor_loss(self, states):
        # Update the actor network
        actor_network_variables = self.model_actor_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_network_variables)

            next_action = self.model_actor_network(states)
            actor_loss = -tf.reduce_mean(self.model_critic_network([states, next_action]))
        actor_grad = tape.gradient(actor_loss, actor_network_variables)
        self.actor_optimiser.apply_gradients(zip(actor_grad, actor_network_variables))
        return actor_loss

    def _save(self, location: str = 'training/results/checkpoints/'):
        # Set the location to save the model and setup the directory
        path = f'{os.getcwd()}/{location}/{self.save_folder}/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the actor and critic model network weights to the path
        self.model_actor_network.save_weights(f'{path}/{self.name.replace(" ", "_")}_actor')
        self.model_critic_network.save_weights(f'{path}/{self.name.replace(" ", "_")}_critic')


class TaskPricingDdpgAgent(DdpgAgent, TaskPricingRLAgent):
    """
    Task pricing ddpg agent
    """

    def __init__(self, agent_name: Union[int, str], actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 min_value: float = 0, max_value: float = 0, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        DdpgAgent.__init__(self, actor_network, critic_network, min_value=min_value, max_value=max_value, **kwargs)
        name = f'Task pricing Ddpg agent {agent_name}' if type(agent_name) is int else agent_name
        TaskPricingRLAgent.__init__(self, name, **kwargs)

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int,
                    training: bool = False):
        observation = tf.expand_dims(self._network_obs(auction_task, allocated_tasks, server, time_step), axis=0)
        action = self.model_actor_network(observation)
        if training:
            return max(0.0, action + tf.random.normal(action.shape, 0, self.epsilon_std))
        else:
            return action


class ResourceWeightingDdpgAgent(DdpgAgent, ResourceWeightingRLAgent):
    """
    Resource weighting ddpg agent
    """

    def __init__(self, agent_name: Union[int, str], actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 min_value: float = -20, max_value: float = 15, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        DdpgAgent.__init__(self, actor_network, critic_network, min_value=min_value, max_value=max_value, **kwargs)
        name = f'Resource weighting Ddpg agent {agent_name}' if type(agent_name) is int else agent_name
        ResourceWeightingRLAgent.__init__(self, name, **kwargs)

    def _get_actions(self, tasks: List[Task], server: Server, time_step: int,
                     training: bool = False) -> Dict[Task, float]:
        observations = tf.convert_to_tensor([self._network_obs(task, tasks, server, time_step) for task in tasks],
                                            dtype='float32')
        actions = self.model_actor_network(observations)
        if training:
            actions += tf.random.normal(actions.shape, 0, self.epsilon_std)
        return {task: max(0.0, float(action)) for task, action in zip(tasks, actions)}


class TD3Agent(DdpgAgent, ABC):
    """
    Twin-delayed ddpg agent
    """

    def __init__(self, actor_network: tf.keras.Model, critic_network: tf.keras.Model, twin_critic_network: tf.keras.Model,
                 twin_critic_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 actor_update_frequency: int = 2, **kwargs):
        DdpgAgent.__init__(self, actor_network, critic_network, **kwargs)

        # Twin critic
        assert id(critic_network) != id(twin_critic_network) and twin_critic_network.output_shape[-1] == 1
        self.twin_model_critic_network = twin_critic_network
        self.twin_target_critic_network = tf.keras.models.clone_model(twin_critic_network)
        self.twin_critic_optimiser = twin_critic_optimiser

        # Training attributes
        self.actor_update_frequency = actor_update_frequency

    def _train(self, states, actions, next_states, rewards, dones) -> float:
        rewards, dones = tf.expand_dims(rewards, axis=1), tf.expand_dims(dones, axis=1)
        # Update the critic network
        critic_network_variables = self.model_critic_network.trainable_variables
        twin_critic_network_variables = self.twin_model_critic_network.trainable_variables
        with tf.GradientTape(persistent=True) as critic_tape:
            critic_tape.watch(critic_network_variables)
            critic_tape.watch(twin_critic_network_variables)

            # Calculate the state and next state q values with the actions and the actor next actions
            obs = [states, tf.expand_dims(actions, axis=1)]
            critic_state_q_values = self.model_critic_network(obs)
            twin_critic_state_q_values = self.twin_model_critic_network(obs)

            # Calculate the target using the rewards, discount factor, next q values and dones
            next_actions = self.model_actor_network(next_states) + tf.random.normal((self.batch_size, 1), 0, 0.1)
            next_state_q_values = tf.reduce_min([self.target_critic_network([next_states, next_actions]),
                                                 self.twin_target_critic_network([next_states, next_actions])], axis=0)
            td_target = tf.stop_gradient(rewards + self.discount_factor * next_state_q_values * dones)

            # Calculate the element wise loss
            critic_loss = self.error_loss_fn(td_target, critic_state_q_values)
            twin_critic_loss = self.error_loss_fn(td_target, twin_critic_state_q_values)

        # Find the critic and twin critic gradients and update the networks then delete the tape
        critic_grads = critic_tape.gradient(critic_loss, critic_network_variables)
        twin_critic_grads = critic_tape.gradient(twin_critic_loss, twin_critic_network_variables)
        del critic_tape
        self.critic_optimiser.apply_gradients(zip(critic_grads, critic_network_variables))
        self.twin_critic_optimiser.apply_gradients(zip(twin_critic_grads, twin_critic_network_variables))

        if self.total_updates % self.actor_update_frequency == 0:
            # update the actor network
            actor_loss = self._actor_loss(states)
        else:
            actor_loss = 0

        # Check if to update the target, if so update each variable at a time using the target update tau variable
        if self.total_updates % self.actor_target_update_frequency == 0:
            ReinforcementLearningAgent._update_target_network(self.model_actor_network, self.target_actor_network,
                                                              self.target_update_tau)
        if self.total_updates % self.critict_target_update_frequency == 0:
            ReinforcementLearningAgent._update_target_network(self.model_critic_network, self.target_critic_network,
                                                              self.target_update_tau)
            ReinforcementLearningAgent._update_target_network(self.twin_model_critic_network,
                                                              self.twin_target_critic_network, self.target_update_tau)

        return critic_loss + actor_loss

    # noinspection DuplicatedCode
    def _save(self, location: str = 'training/results/checkpoints/'):
        # Set the location to save the model and setup the directory
        path = f'{os.getcwd()}/{location}/{self.save_folder}/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the actor and critic model network weights to the path
        self.model_actor_network.save_weights(f'{path}/{self.name.replace(" ", "_")}_actor')
        self.model_critic_network.save_weights(f'{path}/{self.name.replace(" ", "_")}_critic')
        self.twin_model_critic_network.save_weights(f'{path}/{self.name.replace(" ", "_")}_twin_critic')


class TaskPricingTD3Agent(TD3Agent, TaskPricingDdpgAgent):
    """
    Task pricing twin-delayed ddpg agent
    """

    def __init__(self, agent_num: int, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 twin_critic_network: tf.keras.Model, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        TD3Agent.__init__(self, actor_network, critic_network, twin_critic_network, **kwargs)
        TaskPricingDdpgAgent.__init__(self, f'Task pricing TD3 agent {agent_num}', actor_network, critic_network, **kwargs)


class ResourceWeightingTD3Agent(TD3Agent, ResourceWeightingDdpgAgent):
    """
    Resource weighting twin-delayed ddpg agent
    """

    def __init__(self, agent_name: Union[int, str], actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 twin_critic_network: tf.keras.Model, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        TD3Agent.__init__(self, actor_network, critic_network, twin_critic_network, **kwargs)
        name = f'Resource weighting TD3 agent {agent_name}' if type(agent_name) is int else agent_name
        ResourceWeightingDdpgAgent.__init__(self, name, actor_network, critic_network, **kwargs)
