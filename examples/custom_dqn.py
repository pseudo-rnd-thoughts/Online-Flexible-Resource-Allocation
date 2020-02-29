"""
Custom DQN implementation from
https://github.com/kimmyungsup/Reinforcement-Learning-with-Tensorflow-2.0/blob/master/DQN_tf20/dqn_tf20.py
"""

import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

env = gym.make('CartPole-v0')
num_action = env.action_space.n
state_size = env.observation_space.shape[0]


class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = Dense(64, activation='relu')
        self.layer2 = Dense(64, activation='relu')
        self.value = Dense(num_action)

    def call(self, state, **kwargs):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value


class DQNtrain:
    def __init__(self):
        # hyper parameters
        self.lr = 0.001
        self.lr2 = 0.001
        self.df = 0.99

        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = optimizers.RMSprop(lr=self.lr, )

        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.state_size = state_size

        self.memory = deque(maxlen=2000)

        # tensorboard
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(num_action)
        else:
            q_value = self.dqn_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            target = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            target_val = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))

            target = np.array(target)
            target_val = np.array(target_val)

            for i in range(self.batch_size):
                # noinspection PyArgumentList
                next_v = np.array(target_val[i]).max()
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + self.df * next_v

            values = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            error = tf.square(values - target) * 0.5
            error = tf.reduce_mean(error)

        dqn_grads = tape.gradient(error, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

    def run(self):

        t_end = 500
        epi = 100000

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for e in range(epi):
            total_reward = 0
            for t in range(t_end):
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                # env.render()

                if t == t_end:
                    done = True
                if t < t_end and done:
                    reward = -1

                total_reward += reward
                self.append_sample(state, action, reward, next_state, done)

                if len(self.memory) >= self.train_start:
                    self.train()

                total_reward += reward
                state = next_state

                if done:
                    self.update_target()
                    self.reward_board(total_reward)
                    print("e : ", e, " reward : ", total_reward, " step : ", t)
                    env.reset()
                    with self.train_summary_writer.as_default():
                        # tf.summary.scalar('actor_loss', self.train_loss.result(), step=e)
                        tf.summary.scalar('reward', total_reward, step=e)
                    break


if __name__ == '__main__':
    DQN = DQNtrain()
    DQN.run()
