import random as rnd

import gym
import tensorflow as tf

from agents.reinforcement_learning_agent.dqn import DqnAgent
from agents.reinforcement_learning_agent.policy import EpsilonGreedyPolicy, GreedyPolicy


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

    network = tf.keras.Sequential([
        tf.keras.layers.Input(env.observation_space.shape),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1()),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1()),
        tf.keras.layers.Dense(env.action_space.n, activation='linear', kernel_regularizer=tf.keras.regularizers.l1())
    ])
    dqn_agent = DqnAgent(network)
    training_policy, eval_policy = EpsilonGreedyPolicy(dqn_agent), GreedyPolicy(dqn_agent)

    state = env.reset()
    for _ in range(100):
        action = rnd.randint(0, env.action_space.n - 1)
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
            state = env.reset()


if __name__ == "__main__":
    train()
