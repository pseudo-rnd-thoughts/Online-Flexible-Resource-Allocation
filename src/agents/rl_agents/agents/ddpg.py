"""
Continuous action space policy gradient agents
"""

import os
import random as rnd
from abc import ABC
from typing import List, Dict, Union

import tensorflow as tf

from agents.rl_agents.rl_agents import ReinforcementLearningAgent, TaskPricingRLAgent, ResourceWeightingRLAgent, \
    ResourceAllocationState
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


class DdpgAgent(ReinforcementLearningAgent, ABC):
    """
    Deep deterministic policy gradient agent
    """

    def __init__(self, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 actor_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.RMSprop(lr=0.0001),
                 critic_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.RMSprop(lr=0.0005),
                 initial_epsilon_std: float = 4.5, final_epsilon_std: float = 0.5, epsilon_steps: int = 20000,
                 epsilon_update_freq: int = 250, epsilon_log_freq: int = 2000, min_value: float = -15.0,
                 max_value: float = 15.0, target_update_tau: float = 0.01, actor_target_update_freq: int = 1,
                 critic_target_update_freq: int = 1, upper_action_bound: float = 30.0, **kwargs):
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
        self.actor_target_update_freq = actor_target_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.upper_action_bound = upper_action_bound

        # Exploration
        self.initial_epsilon_std = initial_epsilon_std
        self.final_epsilon_std = final_epsilon_std
        self.epsilon_steps = epsilon_steps
        self.epsilon_std = initial_epsilon_std

        self.epsilon_update_freq = epsilon_update_freq
        self.epsilon_log_freq = epsilon_log_freq

    def _update_epsilon(self):
        self.total_actions += 1

        if self.total_actions % self.epsilon_update_freq == 0:
            self.epsilon_std = max((
                                               self.final_epsilon_std - self.initial_epsilon_std) * self.total_actions / self.epsilon_steps + self.initial_epsilon_std,
                                   self.final_epsilon_std)

            if self.total_actions % self.epsilon_log_freq == 0:
                tf.summary.scalar(f'{self.name} agent epsilon std', self.epsilon_std, self.total_actions)
                tf.summary.scalar('Epsilon std', self.epsilon_std, self.total_actions)

    def _train(self, states, actions, next_states, rewards, dones) -> float:
        # The rewards and dones dims need to be expanded for the td_target to have the same shape as the q values
        rewards, dones = tf.expand_dims(rewards, axis=1), tf.expand_dims(dones, axis=1)

        # Critic loss
        critic_loss = self._critic_loss(states, actions, next_states, rewards, dones)

        # Actor loss
        actor_loss = self._actor_loss(states)

        # Check if to update the target, if so update each variable at a time using the target update tau variable
        if self.total_updates % self.actor_target_update_freq == 0:
            self._update_target_network(self.model_actor_network, self.target_actor_network, self.target_update_tau)
        if self.total_updates % self.critic_target_update_freq == 0:
            self._update_target_network(self.model_critic_network, self.target_critic_network, self.target_update_tau)

        return critic_loss + actor_loss

    def _actor_loss(self, states):
        # Update the actor network
        actor_network_variables = self.model_actor_network.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_network_variables)

            next_action = tf.clip_by_value(self.model_actor_network(states), 0, self.upper_action_bound)
            actor_loss = -tf.reduce_mean(self.model_critic_network([states, next_action]))
        actor_grad = tape.gradient(actor_loss, actor_network_variables)
        self.actor_optimiser.apply_gradients(zip(actor_grad, actor_network_variables))
        return actor_loss

    def _critic_loss(self, states, actions, next_states, rewards, dones):
        # Update the critic network
        critic_network_variables = self.model_critic_network.trainable_variables
        with tf.GradientTape() as critic_tape:
            critic_tape.watch(critic_network_variables)

            # Calculate the state and next state q values with the actions and the actor next actions
            state_q_values = self.model_critic_network([states, tf.expand_dims(actions, axis=1)])
            next_actions = tf.clip_by_value(self.model_actor_network(next_states), 0, self.upper_action_bound)
            next_state_q_values = self.target_critic_network([next_states, next_actions])

            # Calculate the target using the rewards, discount factor, next q values and dones
            td_target = tf.stop_gradient(rewards + self.discount_factor * next_state_q_values * dones)

            # Calculate the element wise loss
            critic_loss = self.error_loss_fn(td_target, state_q_values)
        critic_grads = critic_tape.gradient(critic_loss, critic_network_variables)
        clipped_critic_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in critic_grads]
        self.critic_optimiser.apply_gradients(zip(clipped_critic_grads, critic_network_variables))

        return critic_loss

    # noinspection DuplicatedCode
    def save(self, location: str = 'training/results/checkpoints/'):
        """
        Saves the DDPG agent networks, both the actor and the critic

        Args:
            location: Custom save location
        """
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
                 min_value: float = -100.0, max_value: float = 100.0, epsilon_steps=140000, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        DdpgAgent.__init__(self, actor_network, critic_network, min_value=min_value, max_value=max_value,
                           epsilon_steps=epsilon_steps, **kwargs)
        name = f'Task pricing Ddpg agent {agent_name}' if type(agent_name) is int else agent_name
        TaskPricingRLAgent.__init__(self, name, **kwargs)

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int,
                    training: bool = False):
        observation = tf.expand_dims(self._network_obs(auction_task, allocated_tasks, server, time_step), axis=0)
        action = self.model_actor_network(observation)

        if training:
            self._update_epsilon()
            return float(tf.clip_by_value(action + tf.random.gamma(action.shape, 1, self.epsilon_std),
                                          0.0, self.upper_action_bound))
        else:
            return action


class ResourceWeightingDdpgAgent(DdpgAgent, ResourceWeightingRLAgent):
    """
    Resource weighting ddpg agent
    """

    def __init__(self, agent_name: Union[int, str], actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 min_value: float = -35, max_value: float = 25, epsilon_steps=90000, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        DdpgAgent.__init__(self, actor_network, critic_network, min_value=min_value, max_value=max_value,
                           epsilon_steps=epsilon_steps, **kwargs)
        name = f'Resource weighting Ddpg agent {agent_name}' if type(agent_name) is int else agent_name
        ResourceWeightingRLAgent.__init__(self, name, **kwargs)

    # noinspection DuplicatedCode
    def _get_actions(self, tasks: List[Task], server: Server, time_step: int,
                     training: bool = False) -> Dict[Task, float]:
        observations = tf.cast([self._network_obs(task, tasks, server, time_step) for task in tasks], tf.float32)
        actions = self.model_actor_network(observations)

        if training:
            self._update_epsilon()
            actions += tf.random.gamma(actions.shape, 1, self.epsilon_std)

        clipped_actions = tf.clip_by_value(actions, 0.0, self.upper_action_bound)
        return {task: float(action) for task, action in zip(tasks, clipped_actions)}


class TD3Agent(DdpgAgent, ABC):
    """
    Twin-delayed ddpg agent
    """

    def __init__(self, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 twin_critic_network: tf.keras.Model,
                 twin_critic_optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 actor_update_frequency: int = 3, actor_target_update_frequency: int = 3, **kwargs):
        DdpgAgent.__init__(self, actor_network, critic_network,
                           actor_target_update_frequency=actor_target_update_frequency, **kwargs)

        # Twin critic
        assert id(critic_network) != id(twin_critic_network) and twin_critic_network.output_shape[-1] == 1
        self.twin_model_critic_network = twin_critic_network
        self.twin_target_critic_network = tf.keras.models.clone_model(twin_critic_network)
        self.twin_critic_optimiser = twin_critic_optimiser

        # Training attributes
        self.actor_update_frequency = actor_update_frequency

    def _train(self, states, actions, next_states, rewards, dones) -> float:
        rewards, dones = tf.expand_dims(rewards, axis=1), tf.expand_dims(dones, axis=1)

        # Critic loss
        critic_loss = self._twin_critic_loss(states, actions, next_states, rewards, dones)

        if self.total_updates % self.actor_update_frequency == 0:
            # update the actor network
            actor_loss = self._actor_loss(states)
        else:
            actor_loss = 0

        # Check if to update the target, if so update each variable at a time using the target update tau variable
        if self.total_updates % self.actor_target_update_freq == 0:
            ReinforcementLearningAgent._update_target_network(self.model_actor_network, self.target_actor_network,
                                                              self.target_update_tau)
        if self.total_updates % self.critic_target_update_freq == 0:
            self._update_target_network(self.model_critic_network, self.target_critic_network, self.target_update_tau)
            self._update_target_network(self.twin_model_critic_network, self.twin_target_critic_network,
                                        self.target_update_tau)

        return critic_loss + actor_loss

    def _twin_critic_loss(self, states, actions, next_states, rewards, dones):
        # Update the critic network
        critic_network_variables = self.model_critic_network.trainable_variables
        twin_critic_network_variables = self.twin_model_critic_network.trainable_variables
        with tf.GradientTape(persistent=True) as critic_tape:
            critic_tape.watch(critic_network_variables)
            critic_tape.watch(twin_critic_network_variables)

            # Calculate the state and next state q values with the actions and the actor next actions
            obs = [states, tf.expand_dims(actions, axis=-1)]
            critic_state_q_values = self.model_critic_network(obs)
            twin_critic_state_q_values = self.twin_model_critic_network(obs)

            # Calculate the target using the rewards, discount factor, next q values and dones
            next_actions = self.model_actor_network(next_states) + tf.random.normal((self.batch_size, 1), 0, 0.2)
            clipped_next_actions = tf.clip_by_value(next_actions, 0, self.upper_action_bound)
            next_state_q_values = tf.reduce_min([self.target_critic_network([next_states, clipped_next_actions]),
                                                 self.twin_target_critic_network([next_states, clipped_next_actions])],
                                                axis=0)
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

        return (critic_loss + twin_critic_loss) / 2

    # noinspection DuplicatedCode
    def save(self, location: str = 'training/results/checkpoints/'):
        """
        Saves the TD3 agent networks, both the actor, critic and twin critic

        Args:
            location: Custom save location
        """
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
        TaskPricingDdpgAgent.__init__(self, f'Task pricing TD3 agent {agent_num}', actor_network, critic_network,
                                      **kwargs)


class ResourceWeightingTD3Agent(TD3Agent, ResourceWeightingDdpgAgent):
    """
    Resource weighting twin-delayed ddpg agent
    """

    def __init__(self, agent_name: int, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 twin_critic_network: tf.keras.Model, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width

        TD3Agent.__init__(self, actor_network, critic_network, twin_critic_network, **kwargs)
        ResourceWeightingDdpgAgent.__init__(self, f'Resource weighting TD3 agent {agent_name}', actor_network,
                                            critic_network, **kwargs)


class ResourceWeightingSeq2SeqAgent(TD3Agent, ResourceWeightingRLAgent):
    """
    Resource Weighting Seq2Seq Agent
    """

    network_obs_width = 8

    def __init__(self, agent_num: int, actor_network: tf.keras.Model, critic_network: tf.keras.Model,
                 twin_critic_network: tf.keras.Model, **kwargs):
        assert actor_network.input_shape[-1] == self.network_obs_width
        assert actor_network.output_shape[-1] == 1

        TD3Agent.__init__(self, actor_network, critic_network, twin_critic_network, **kwargs)
        ResourceWeightingRLAgent.__init__(self, f'Resource Weighting Seq2Seq agent {agent_num}', **kwargs)

    # noinspection DuplicatedCode
    def _get_actions(self, tasks: List[Task], server: Server, time_step: int,
                     training: bool = False) -> Dict[Task, float]:
        observations = [self._normalise_task(task, server, time_step) for task in tasks]
        actions = self.model_actor_network(tf.cast(tf.expand_dims(observations, 0), tf.float32))[0]

        if training:
            self._update_epsilon()
            actions += tf.random.gamma(actions.shape, 1, self.epsilon_std)

        clipped_actions = tf.clip_by_value(actions, 0.0, self.upper_action_bound)
        return {task: float(action) for task, action in zip(tasks, clipped_actions)}

    def train(self):
        """
        Trains the reinforcement learning agent and logs the training loss
        Due to the actions being a list, the actions must be padded
        """
        states, actions, next_states, rewards, dones = zip(*rnd.sample(self.replay_buffer, self.batch_size))

        states = tf.keras.preprocessing.sequence.pad_sequences(list(states), dtype='float32')
        actions = tf.keras.preprocessing.sequence.pad_sequences(list(actions), dtype='float32')
        next_states = tf.keras.preprocessing.sequence.pad_sequences(list(next_states), dtype='float32')
        rewards = tf.cast(tf.stack(rewards), tf.float32)
        dones = tf.cast(tf.stack(dones), tf.float32)

        training_loss = self._train(states, actions, next_states, rewards, dones)
        if self.total_updates % self.training_loss_log_freq == 0:
            tf.summary.scalar(f'{self.name} agent training loss', training_loss, self.total_observations)
            tf.summary.scalar(f'Training loss', training_loss, self.total_observations)
        if self.total_updates % self.save_frequency == 0:
            self.save()
        self.total_updates += 1

    def resource_allocation_obs(self, agent_state: ResourceAllocationState, actions: Dict[Task, float],
                                next_agent_state: ResourceAllocationState, finished_tasks: List[Task]):
        """
        Resource allocation observation

        Args:
            agent_state: The agent state
            actions: Dictionary of agent actions
            next_agent_state: The next agent state
            finished_tasks: List of finished tasks
        """
        if len(agent_state.tasks) <= 1 or len(next_agent_state.tasks) <= 1:
            return

        reward = sum(self.success_reward if finished_task.stage is TaskStage.COMPLETED else self.failed_reward
                     for finished_task in finished_tasks)
        obs = tf.cast([self._normalise_task(task, agent_state.server, agent_state.time_step)
                       for task in agent_state.tasks], tf.float32)
        next_obs = tf.cast([self._normalise_task(task, next_agent_state.server, next_agent_state.time_step)
                            for task in next_agent_state.tasks], tf.float32)
        task_actions = [actions[task] for task in agent_state.tasks]
        # noinspection PyTypeChecker
        # Due to the action expected as a float not List[float]
        self._add_trajectory(obs, task_actions, next_obs, reward)
