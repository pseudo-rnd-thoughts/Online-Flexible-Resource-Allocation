import tensorflow as tf

if __name__ == "__main__":
    other_tasks_input_layer = tf.keras.layers.Input(shape=(None, 8), name='Other_tasks')
    auction_task_input_layer = tf.keras.layers.Input(shape=(8,), name='Auction_task')

    lstm_layer = tf.keras.layers.LSTM(64)(other_tasks_input_layer)
    concat_layer = tf.keras.layers.concatenate([lstm_layer, auction_task_input_layer])
    relu_layer = tf.keras.layers.Dense(32, activation='relu', name='Relu')(concat_layer)
    q_value_layer = tf.keras.layers.Dense(11, activation='linear', name='Q_value')(relu_layer)

    task_pricing = tf.keras.Model(inputs=[other_tasks_input_layer, auction_task_input_layer], outputs=q_value_layer,
                                  name='TaskPricing')

    tf.keras.utils.plot_model(task_pricing, to_file='../../final_report/figures/task_pricing_network_architecture.png',
                              expand_nested=True)

    other_tasks_input_layer = tf.keras.layers.Input(shape=(None, 8), name='Other_tasks')
    weighting_task_input_layer = tf.keras.layers.Input(shape=(8,), name='Weighting_task')

    lstm_layer = tf.keras.layers.LSTM(64)(other_tasks_input_layer)
    concat_layer = tf.keras.layers.concatenate([lstm_layer, weighting_task_input_layer])
    relu_layer = tf.keras.layers.Dense(32, activation='relu', name='Relu')(concat_layer)
    q_value_layer = tf.keras.layers.Dense(11, activation='linear', name='Q_value')(relu_layer)

    weighting_net = tf.keras.Model(inputs=[other_tasks_input_layer, weighting_task_input_layer], outputs=q_value_layer,
                                   name='Resource_Weighting')

    tf.keras.utils.plot_model(weighting_net,
                              to_file='../../final_report/figures/resource_weighting_network_architecture.png',
                              expand_nested=True)
