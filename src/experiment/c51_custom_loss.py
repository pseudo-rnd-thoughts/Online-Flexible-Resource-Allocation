import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.agents import CategoricalDqnAgent
from tf_agents.agents.categorical_dqn.categorical_dqn_agent import project_distribution
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.categorical_q_network import CategoricalQNetwork
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


# @tf.function
def train_agent(dqn_agent, trajectories, gamma=1.0):
    network_variables = dqn_agent._q_network.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(network_variables)

        time_steps, actions, next_time_steps = dqn_agent._experience_to_transitions(trajectories)

        # q_logits contains the Q-value logits for all actions.
        q_logits, _ = dqn_agent._q_network(time_steps.observation, time_steps.step_type)
        next_q_distribution = dqn_agent._next_q_distribution(next_time_steps)

        # Project the sample Bellman update \hat{T}Z_{\theta} onto the original
        # support of Z_{\theta} (see Figure 1 in paper).
        tiled_support = tf.ones((batch_size, dqn_agent._num_atoms)) * dqn_agent._support

        discount = tf.expand_dims(next_time_steps.discount, axis=-1)
        reward = tf.expand_dims(next_time_steps.reward, axis=-1)
        target_support = reward + gamma * discount * tiled_support

        v_min, v_max, delta_z = -10.0, 10.0, (dqn_agent._support[1:] - dqn_agent._support[:-1])[0]
        # print(f'V min: {v_min}, V max: {v_max}, Delta z: {delta_z}, Num atoms: {num_atoms}\n')
        # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
        clipped_support = tf.expand_dims(tf.clip_by_value(target_support, v_min, v_max), axis=1)
        # print(f'Clipped support: {clipped_support}')
        tiled_support = tf.tile([clipped_support], [1, 1, num_atoms, 1])
        # print(f'Tiled support: {tiled_support}')
        reshaped_target_support = tf.reshape(tf.ones([batch_size, 1]) * agent._support, [batch_size, num_atoms, 1])
        # print(f'Reshaped target support: {reshaped_target_support}')
        # numerator = `|clipped_support - z_i|` in Eq7.
        numerator = tf.abs(tiled_support - reshaped_target_support)
        # print(f'Numerator: {numerator}')
        quotient = 1 - (numerator / delta_z)
        # print(f'Quotient: {quotient}')
        clipped_quotient = tf.clip_by_value(quotient, 0, 1)
        # print(f'Clipped quotient: {clipped_quotient}')
        # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
        inner_prod = clipped_quotient * tf.expand_dims(next_q_distribution, axis=1)
        # print(f'Inner prod: {inner_prod}')
        projection = tf.reduce_sum(inner_prod, 3)[0]
        # print(f'Projection: {projection}')

        target_projection = project_distribution(target_support, next_q_distribution, agent._support)
        target_distribution = tf.stop_gradient(projection)
        assert tf.reduce_all(target_projection == target_distribution), f'{target_projection}\n\n{target_distribution}'

        # Obtain the current Q-value logits for the selected actions.
        reshaped_actions = tf.stack([tf.range(batch_size, dtype=tf.int64), actions], axis=-1)
        chosen_action_logits = tf.gather_nd(q_logits, reshaped_actions)

        critic_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(target_distribution, chosen_action_logits))

    grads = tape.gradient(critic_loss, network_variables)
    dqn_agent._optimizer.apply_gradients(zip(grads, network_variables))
    dqn_agent._update_target()

    return critic_loss


if __name__ == "__main__":
    batch_size = 64
    num_atoms = 51
    env_name = 'CartPole-v0'

    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)

    train_env = TFPyEnvironment(train_py_env)
    eval_env = TFPyEnvironment(eval_py_env)

    q_net = CategoricalQNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=(100,),
                                num_atoms=num_atoms)
    optimiser = tf.keras.optimizers.Adam(lr=1e-3)
    agent = CategoricalDqnAgent(train_env.time_step_spec(), train_env.action_spec(), categorical_q_network=q_net,
                                optimizer=optimiser, td_errors_loss_fn=common.element_wise_squared_loss)
    agent.initialize()

    eval_policy, training_policy = agent.policy, agent.collect_policy
    random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=train_env.batch_size,
                                          max_length=10000)
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    for _ in range(100):
        training_step(train_env, random_policy, replay_buffer)

    agent.train = common.function(agent.train)
    eval_avg_rewards = [eval_agent(eval_env, eval_policy)]

    experience, _ = next(iterator)
    train_agent(agent, experience)

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
