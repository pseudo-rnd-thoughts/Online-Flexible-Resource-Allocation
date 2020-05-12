"""
Allows for the visualisation of the auction and resource weighting networks
"""

import tensorflow as tf

if __name__ == "__main__":
    folder = '../../../final_report/figures/3_solution_figs'
    # --------------------------------------------------------------------------------------
    # Task pricing network
    other_tasks_input_layer = tf.keras.layers.Input(shape=(None, 8), name='Other_Tasks')
    auction_task_input_layer = tf.keras.layers.Input(shape=(8,), name='Auction_Task')

    lstm_layer = tf.keras.layers.LSTM(32, name='Encode')(other_tasks_input_layer)
    concat_layer = tf.keras.layers.concatenate([lstm_layer, auction_task_input_layer], name='Combine_Tasks')
    relu_layer = tf.keras.layers.Dense(16, activation='relu', name='ReLU')(concat_layer)
    q_value_layer = tf.keras.layers.Dense(21, activation='linear', name='Q_Values')(relu_layer)
    actor_layer = tf.keras.layers.Dense(1, activation='relu', name='Actor')(relu_layer)

    task_pricing = tf.keras.Model(inputs=[other_tasks_input_layer, auction_task_input_layer],
                                  outputs=[q_value_layer, actor_layer], name='Task_Pricing')
    tf.keras.utils.plot_model(task_pricing, to_file=f'{folder}/task_pricing_network_architecture.png',
                              expand_nested=True)

    # -------------------------------------------------------------------------------------
    # Single task resource weighting network
    other_tasks_input_layer = tf.keras.layers.Input(shape=(None, 8), name='Other_Tasks')
    weighting_task_input_layer = tf.keras.layers.Input(shape=(8,), name='Weighting_Task')

    lstm_layer = tf.keras.layers.LSTM(32, name='Encode')(other_tasks_input_layer)
    concat_layer = tf.keras.layers.concatenate([lstm_layer, weighting_task_input_layer], name='Combine_Tasks')
    relu_layer = tf.keras.layers.Dense(16, activation='relu', name='ReLU')(concat_layer)
    q_value_layer = tf.keras.layers.Dense(11, activation='linear', name='Q_Values')(relu_layer)
    actor_layer = tf.keras.layers.Dense(1, activation='linear', name='Actor')(relu_layer)

    weighting_net = tf.keras.Model(inputs=[other_tasks_input_layer, weighting_task_input_layer],
                                   outputs=[q_value_layer, actor_layer], name='Resource_Weighting')

    print(f'Saving single task weighting network')
    tf.keras.utils.plot_model(weighting_net, to_file=f'{folder}/single_task_weighting_network_architecture.png',
                              expand_nested=True)

    # --------------------------------------------------------------------------------------
    # Seq2Seq actor
    input_layer = tf.keras.layers.Input(shape=(None, 8), name='Tasks')

    encoder = tf.keras.layers.LSTM(32, return_state=True, name='Encoder')
    encoder_output, encoder_state_h, encoder_state_c = encoder(input_layer)  # Ignore the encoder_output

    decoder = tf.keras.layers.LSTM(32, return_sequences=True, name='Decoder')
    decoded = decoder(input_layer, initial_state=[encoder_state_h, encoder_state_c])
    actor_layer = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(),
                                        name='Actor')(decoded)

    actor_network = tf.keras.Model(name='Seq2Seq_Actor', inputs=input_layer, outputs=actor_layer)

    print(f'Saving actor network')
    tf.keras.utils.plot_model(actor_network, to_file=f'{folder}/multi_task_actor_weighting_network_architecture.png',
                              expand_nested=True)

    # Seq2Seq critic
    task_input_layer = tf.keras.layers.Input(shape=(None, 8), name='Tasks')
    action_input_layer = tf.keras.layers.Input(shape=(None, 1), name='Actions')

    concat_layer = tf.keras.layers.concatenate([task_input_layer, action_input_layer], name='Concatenated_Task_Actions')
    lstm_layer = tf.keras.layers.LSTM(32, name='Task_Action_Encoder')(concat_layer)
    relu_layer = tf.keras.layers.Dense(16, activation='relu', name='ReLU')(lstm_layer)
    q_value = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l1(),
                                    name='Q_Value')(relu_layer)

    critic_network = tf.keras.Model(name='Seq2Seq_Critic', inputs=[task_input_layer, action_input_layer],
                                    outputs=q_value)

    print(f'Saving critic network')
    tf.keras.utils.plot_model(critic_network, to_file=f'{folder}/single_task_critic_weighting_network_architecture.png',
                              expand_nested=True)
