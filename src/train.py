import gym
import tensorflow as tf

from agents.dqn import DqnAgent
from agents.random import RandomAgent
from policy import GreedyPolicy, EpsilonGreedyPolicy


def eval_agent(policy, env):
    rewards = 0
    for test_num in range(10):
        state = env.reset()
        for step in range(200):
            action = policy.action(state)
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
    return rewards / 10


def train():
    env = gym.make('CartPole-v1')

    random_agent = RandomAgent(env.action_space.n)

    network = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(),
                              kernel_initializer=tf.keras.initializers.glorot_uniform(),
                              input_shape=(env.observation_space,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(),
                              kernel_initializer=tf.keras.initializers.glorot_uniform()),
        tf.keras.layers.Dense(env.action_space.n, activation='linear', kernel_regularizer=tf.keras.regularizers.l1(),
                              kernel_initializer=tf.keras.initializers.glorot_uniform())
    ])
    dqn_agent = DqnAgent('dqn', network)
    eval_policy = GreedyPolicy(dqn_agent)
    training_policy = EpsilonGreedyPolicy(dqn_agent, epsilon=0.2)

    state = env.reset()
    for _ in range(100):
        action = random_agent.action()
        next_state, reward, done, info = env.step(action)
        dqn_agent.observation(state, action, next_state, reward, done)

        if done:
            state = env.reset()
        else:
            state = next_state

    state = env.reset()
    for step in range(20000):
        action = training_policy.action(state)
        next_state, reward, done, info = env.step(action)

        dqn_agent.observation(state, action, next_state, reward, done)

        if done:
            state = env.reset()
        else:
            state = next_state

        loss = dqn_agent.train()

        if step % 200 == 0:
            print(f'Step: {step}, loss: {loss}')
        if step % 1000 == 0:
            avg_rewards = eval_agent(eval_policy, env)
            print(f'Eval step: {step}, avg rewards: {avg_rewards}')


if __name__ == "__main__":
    train()
