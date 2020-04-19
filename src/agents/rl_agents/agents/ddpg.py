"""
Continuous action space policy gradient agents
"""

import os
from abc import ABC
from typing import List, Dict, Union

import numpy as np
import tensorflow as tf

from agents.rl_agents.rl_agents import ReinforcementLearningAgent, TaskPricingRLAgent, ResourceWeightingRLAgent
from env.server import Server
from env.task import Task


# Todo agents: TD3, D4PG, MADDPG, Seq2Seq DDPG


class DdpgAgent(ReinforcementLearningAgent, ABC):
    """
    Deep deterministic policy gradient agent
    """

    def __init__(self, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 actor_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 critic_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 initial_epsilon_std: float = 0.8, final_epsilon_std: float = 0.1, epsilon_steps: int = 10000,
                 epsilon_update_frequency: int = 25, min_value: float = -15.0, max_value: float = 15, **kwargs):
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
        # Update the critic network
        critic_network_variables = self.model_critic_network.trainable_variables
        with tf.GradientTape() as critic_tape:
            critic_tape.watch(critic_network_variables)

            # Calculate the state and next state q values with the actions and the actor next actions
            state_q_values = self.model_critic_network([np.array(states), np.array(actions)])
            next_actions = self.model_actor_network(next_states)
            next_state_q_values = self.target_critic_network([np.array(next_states), np.array(next_actions)])

            # Calculate the target using the rewards, discount factor, next q values and dones
            td_target = tf.stop_gradient(rewards + self.discount_factor * next_state_q_values * dones)
            td_target = tf.clip_by_value(td_target, self.min_value, self.max_value)

            # Calculate the element wise loss
            critic_loss = self.error_loss_fn(td_target, state_q_values)
        critic_grads = critic_tape.gradient(critic_loss, critic_network_variables)
        self.optimiser.apply_gradients(zip(critic_grads, critic_network_variables))

        # Update the actor network
        actor_network_variables = self.model_actor_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_network_variables)

            next_action = self.model_actor_network(states)
            actor_loss = -tf.reduce_mean(self.model_critic_network([np.array(states), np.array(next_action)]))
        actor_grad = tape.gradient(actor_loss, actor_network_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_network_variables))

        # Check if to update the target, if so update each variable at a time using the target update tau variable
        if self.total_updates % self.actor_update_frequency == 0:
            ReinforcementLearningAgent._update_target_network(self.model_actor_network, self.target_actor_network,
                                                              self.target_update_tau)
        if self.total_updates % self.critic_update_frequency == 0:
            ReinforcementLearningAgent._update_target_network(self.model_critic_network, self.target_critic_network,
                                                              self.target_update_tau)

        return critic_loss + actor_loss

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
                 **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        DdpgAgent.__init__(self, actor_network, critic_network, **kwargs)
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
                 **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        DdpgAgent.__init__(self, actor_network, critic_network, **kwargs)
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
