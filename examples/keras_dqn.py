"""
Keras DQN network from https://github.com/llSourcell/deep_q_learning/blob/master/03_PlayingAgent.ipynb
"""

from keras.models import Sequential  # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque  # For storing moves

import numpy as np
import gym  # To train our network
import random  # For sampling batches from the observations

env = gym.make('MountainCar-v0')  # Choose game (any in the gym should work)

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, activation="relu", kernel_initializer="uniform"))
model.add(Flatten())  # Flatten input so as to have no problems with processing
model.add(Dense(18, activation="relu", kernel_initializer="uniform"))
model.add(Dense(10, activation="relu", kernel_initializer="uniform"))
model.add(Dense(env.action_space.n, activation="linear", kernel_initializer="uniform"))  # Same number of outputs as possible actions

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Parameters
D = deque()  # Register where the actions will be stored

observetime = 500  # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7  # Probability of doing a random move
gamma = 0.9  # Discounted future reward. How much we care about steps further in time
mb_size = 50  # Learning minibatch size

# FIRST STEP: Knowing what each action does (Observing)

observation = env.reset()  # Game begins
obs = np.expand_dims(observation,
                     axis=0)  # (Formatting issues) Making the observation the first element of a batch of inputs
state = np.stack((obs, obs), axis=1)
done = False
for t in range(observetime):
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, env.action_space.n, size=1)[0]
    else:
        Q = model.predict(state)  # Q-values predictions
        action = np.argmax(Q)  # Move with highest Q-value is the chosen one
    observation_new, reward, done, info = env.step(action)  # See state of the game, reward... after performing the action
    obs_new = np.expand_dims(observation_new, axis=0)  # (Formatting issues)
    state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)  # Update the input with the new state of the game
    D.append((state, action, reward, state_new, done))  # 'Remember' action and consequence
    state = state_new  # Update state
    if done:
        env.reset()  # Restart game if it's finished
        obs = np.expand_dims(observation, axis=0)  # (Formatting issues) Making the observation the first element of a batch of inputs
        state = np.stack((obs, obs), axis=1)
print('Observing Finished')

# SECOND STEP: Learning from the observations (Experience replay)

minibatch = random.sample(D, mb_size)  # Sample some moves

inputs_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros(inputs_shape)
targets = np.zeros((mb_size, env.action_space.n))

for i in range(0, mb_size):
    state, action, reward, state_new, done = minibatch[i]

    # Build Bellman equation for the Q function
    inputs[i:i + 1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)

    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)

    # Train network to output the Q function
    model.train_on_batch(inputs, targets)
print('Learning Finished')

# THIRD STEP: Play!

observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    # env.render()  # Uncomment to see game running
    Q = model.predict(state)
    action = np.argmax(Q)
    observation, reward, done, info = env.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(reward))
