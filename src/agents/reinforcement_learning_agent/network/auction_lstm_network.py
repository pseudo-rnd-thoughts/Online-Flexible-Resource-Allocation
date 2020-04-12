
import tensorflow as tf
import numpy as np

from env.environment import OnlineFlexibleResourceAllocationEnv, StepType

if __name__ == "__main__":
    network = tf.keras.Sequential([
        tf.keras.layers.LSTM(10, activation='relu', recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(5, activation='linear')
    ])

    env, (server_state, step_type) = OnlineFlexibleResourceAllocationEnv.load_env('auction_test.env')
    assert step_type is StepType.AUCTION

    auction_actions = {}
    for server, state in server_state.items():
        # print(f'Server: {server.name} - {state}')
        q_values = network(tf.expand_dims(state, axis=0))
        action = tf.math.argmax(q_values, axis=1)
        # print(f'Q Values: {q_values}, argmax: {action}')
        auction_actions[server] = float(action)
        print()

    (next_server_state, next_step_type), server_rewards, done = env.step(auction_actions)

    discount_factor = 0.9
    states, next_states, rewards, actions = [], [], [], []
    for server, state in server_state.items():
        states.append(state)
        next_states.append(next_server_state[server])
        rewards.append(rewards[server] if server in rewards else 0)
        actions.append(auction_actions[server])

    batch_size = len(states)
    states = tf.keras.preprocessing.sequence.pad_sequences(states, dtype='float32')
    next_states = tf.keras.preprocessing.sequence.pad_sequences(next_states, dtype='float32')
    rewards = tf.cast(np.stack(rewards), tf.float32)
    actions = tf.cast(np.stack(actions), tf.int32)
    loss_func = tf.keras.losses.Huber()

    network_variables = network.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(network_variables)

        # print(f'States: {states}')
        state_q_values = network(states)
        action_indexes = tf.stack([tf.range(batch_size, dtype=tf.int32), actions], axis=-1)
        state_action_q_values = tf.gather_nd(state_q_values, action_indexes)
        print(f'State action q values: {state_action_q_values}')

        # print(f'Next states: {next_states}')
        next_state_q_values = network(next_states)
        next_actions = tf.math.argmax(next_state_q_values, axis=1, output_type=tf.int32)
        next_action_indexes = tf.stack([tf.range(batch_size, dtype=tf.int32), next_actions], axis=-1)
        next_state_action_q_values = tf.gather_nd(next_state_q_values, next_action_indexes)
        print(f'Next state action q values: {next_state_action_q_values}')

        td_target = rewards + discount_factor * state_action_q_values
        print(f'Td error: {td_target - state_action_q_values}')
        print(f'Td targets: {td_target}')
        loss = loss_func(td_target, state_action_q_values)
        print(f'Loss: {loss}')

    grads = tape.gradient(loss, network_variables)
