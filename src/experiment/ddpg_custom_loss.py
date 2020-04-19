import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.agents import DdpgAgent
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common

from experiment.dqn_custom_loss import training_step, eval_agent


@tf.function
def train_agent(ddpg_agent, trajectories):
    agent_states, agent_actions, agent_next_states = ddpg_agent._experience_to_transitions(trajectories)

    states = agent_states.observation
    actions = tf.cast(agent_actions, tf.float32)
    rewards = agent_next_states.reward
    dones = tf.cast(~agent_states.is_last(), tf.float32)
    next_states = agent_next_states.observation

    trainable_critic_variables = critic_net.trainable_variables
    with tf.GradientTape() as critic_tape:
        critic_tape.watch(trainable_critic_variables)

        q_values, _ = critic_net((states, actions))
        next_actions, _ = actor_net(next_states)
        next_q_values, _ = critic_net((next_states, next_actions))

        td_targets = tf.stop_gradient(rewards + discount_factor * next_q_values * dones)
        critic_loss = tf.reduce_mean(loss_func(td_targets, q_values))
    critic_grads = critic_tape.gradient(critic_loss, trainable_critic_variables)
    critic_optimiser.apply_gradients(zip(critic_grads, trainable_critic_variables))

    trainable_actor_variables = actor_net.trainable_variables
    with tf.GradientTape() as actor_tape:
        next_actions, _ = actor_net(states)
        q_values, _ = critic_net((states, next_actions))
        actor_loss = -tf.reduce_mean(q_values)
    actor_grads = actor_tape.gradient(actor_loss, trainable_actor_variables)
    actor_optimiser.apply_gradients(zip(actor_grads, trainable_actor_variables))

    ddpg_agent._update_target()
    return actor_loss + critic_loss


if __name__ == "__main__":
    tau = 0.99
    batch_size = 64
    discount_factor = 0.95
    loss_func = tf.losses.mean_squared_error

    env_name = 'MountainCarContinuous-v0'

    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)

    train_env = TFPyEnvironment(train_py_env)
    eval_env = TFPyEnvironment(eval_py_env)

    actor_net = ActorNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=(100,))
    actor_optimiser = tf.keras.optimizers.Adam(lr=0.0001)
    critic_net = CriticNetwork((train_env.time_step_spec().observation, train_env.action_spec()),
                               action_fc_layer_params=(32,), joint_fc_layer_params=(16,), observation_fc_layer_params=(32,))
    critic_optimiser = tf.keras.optimizers.Adam(lr=0.005)
    agent = DdpgAgent(train_env.time_step_spec(), train_env.action_spec(), actor_network=actor_net,
                      critic_network=critic_net, actor_optimizer=actor_optimiser, critic_optimizer=critic_optimiser)
    agent.initialize()

    eval_policy, training_policy = agent.policy, agent.collect_policy
    random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=train_env.batch_size,
                                          max_length=10000)
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    for _ in range(200):
        training_step(train_env, random_policy, replay_buffer)

    agent.train = common.function(agent.train)
    eval_avg_rewards = [eval_agent(eval_env, eval_policy)]

    for step in range(20000):
        training_step(train_env, training_policy, replay_buffer)

        experience, _ = next(iterator)
        # training_loss = agent.train(experience).loss
        training_loss = train_agent(agent, experience)

        if step % 2000 == 0:
            eval_reward = eval_agent(eval_env, eval_policy)
            eval_avg_rewards.append(eval_reward)
            print(f'Step: {step} - Eval avg reward: {eval_reward}')
        elif step % 200 == 0:
            print(f'Step: {step} - Training loss: {training_loss}')

    plt.plot(range(0, 20000 + 1, 2000), eval_avg_rewards)
    plt.ylabel('Eval Average Rewards')
    plt.xlabel('Iteration')
    plt.show()
