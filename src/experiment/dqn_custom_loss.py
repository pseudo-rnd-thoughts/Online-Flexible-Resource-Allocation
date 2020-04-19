
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.agents import DqnAgent

from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


def eval_agent(env, policy, num_tests=10):
    total_rewards = 0.0
    for _ in range(num_tests):
        state = env.reset()
        while not state.is_last():
            policy_action = policy.action(state)
            state = env.step(policy_action.action)
            total_rewards += state.reward
    return float(total_rewards) / num_tests


def training_step(env, policy, buffer):
    state = env.current_time_step()
    policy_step = policy.action(state)
    next_state = env.step(policy_step.action)
    buffer.add_batch(trajectory.from_transition(state, policy_step, next_state))


@tf.function
def train_agent(dqn_agent, trajectories):
    agent_states, agent_actions, agent_next_states = dqn_agent._experience_to_transitions(trajectories)

    states = agent_states.observation
    actions = tf.cast(agent_actions, tf.int32)
    rewards = agent_next_states.reward
    dones = tf.cast(~agent_states.is_last(), tf.float32)
    next_states = agent_next_states.observation

    network_variables = agent._q_network.trainable_weights
    with tf.GradientTape() as tape:
        tape.watch(network_variables)

        q_values, _ = agent._q_network(states)
        next_q_values, _ = agent._target_q_network(next_states)
        next_actions = tf.math.argmax(next_q_values, axis=1, output_type=tf.int32)

        action_indexes = tf.stack([tf.range(batch_size, dtype=tf.int32), actions], axis=-1)
        action_q_values = tf.gather_nd(q_values, action_indexes)
        next_action_indexes = tf.stack([tf.range(batch_size, dtype=tf.int32), next_actions], axis=-1)
        next_actions_q_values = tf.gather_nd(next_q_values, next_action_indexes)

        target = rewards + discount_factor * next_actions_q_values * dones
        loss = loss_func(target, action_q_values)

    grads = tape.gradient(loss, network_variables)
    optimiser.apply_gradients(zip(grads, network_variables))

    for target_variable, model_variable in zip(agent._target_q_network.variables, agent._q_network.variables):
        if target_variable.trainable and model_variable.trainable:
            target_variable.assign(tau * target_variable + (1 - tau) * model_variable)

    return loss


if __name__ == "__main__":
    tau = 0.99
    batch_size = 64
    discount_factor = 0.95
    loss_func = tf.losses.mean_squared_error
    env_name = 'CartPole-v0'

    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)

    train_env = TFPyEnvironment(train_py_env)
    eval_env = TFPyEnvironment(eval_py_env)

    q_net = QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=(100,))
    optimiser = tf.keras.optimizers.Adam(lr=1e-3)
    agent = DqnAgent(train_env.time_step_spec(), train_env.action_spec(), q_network=q_net, optimizer=optimiser,
                     td_errors_loss_fn=common.element_wise_squared_loss)
    agent.initialize()

    eval_policy, training_policy = agent.policy, agent.collect_policy
    random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=train_env.batch_size,
                                          max_length=10000)
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    for _ in range(100):
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