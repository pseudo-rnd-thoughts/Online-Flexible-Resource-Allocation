import tensorflow as tf
import numpy as np

from env.environment import OnlineFlexibleResourceAllocationEnv, StepType


class Seq2SeqModel(tf.keras.Model):
    def __init__(self, units, num_actions):
        tf.keras.Model.__init__(self, name='Seq2Seq2 Model')

        self.units = units
        self.encoder_gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                               recurrent_initializer='glorot_uniform')
        self.decoder_gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                               recurrent_initializer='glorot_uniform')
        self.q_values = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs, training=None, mask=None):
        encoder_sequence, encoder_state = self.encoder_gru(inputs)
        # print(f'Encoder sequence: {encoder_sequence}')
        # print(f'Encoder state: {encoder_state}')
        decoder_sequence, decoder_state = self.decoder_gru(encoder_sequence)
        # print(f'Decoder sequence: {decoder_sequence}')
        # print(f'Decoder state: {decoder_state}')
        output = self.q_values(decoder_sequence)
        # print(f'Output: {output}')
        return output


if __name__ == "__main__":
    network = Seq2SeqModel(8, 2)

    env, (server_state, step_type) = OnlineFlexibleResourceAllocationEnv.load_env('settings/resource_test.env')
    assert step_type is StepType.RESOURCE_ALLOCATION

    auction_actions = {}
    for server, state in server_state.items():
        state = tf.expand_dims(state, axis=0)
        # print(f'{server.name} Server: {state}')
        q_values = network(state)
        # print(f'Q Values: {q_values}')
        argmax_actions = tf.math.argmax(q_values, axis=2, output_type=tf.int32)
        auction_actions[server] = [float(action) for action in argmax_actions[0]]

    (next_server_state, next_step_type), server_rewards, done = env.step(auction_actions)

    discount_factor = 0.9
    states, next_states, rewards, actions = [], [], [], []
    for server, state in server_state.items():
        states.append(state)
        next_states.append(next_server_state[server])
        rewards.append(server_rewards[server])
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
