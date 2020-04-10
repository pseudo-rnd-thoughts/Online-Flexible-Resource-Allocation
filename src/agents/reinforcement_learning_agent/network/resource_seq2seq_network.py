import tensorflow as tf


class ResourceSeqNetwork(tf.keras.Model):
    pass


class EncoderModel(tf.keras.Model):
    def __init__(self, encoder_units: int = 64, batch_size: int = 32):
        tf.keras.Model.__init__(self, name='Encoder Model')
        self.encoder_units = encoder_units
        self.batch_size = batch_size

        self.gru = tf.keras.layers.GRU(self.encoder_units, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialise_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units: int):
        tf.keras.layers.Layer.__init__(self, name='Bahdanau Attention')

        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.v(tf.nn.tanh(self.w1(query_with_time_axis) + self.w2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class DecoderModel(tf.keras.Model):
    def __init__(self, decoder_units: int = 64, num_actions: int = 10, batch_size: int = 32):
        tf.keras.Model.__init__(self, name='Decoder Model')

        self.decoder_units = decoder_units
        self.batch_size = batch_size

        self.gru = tf.keras.layers.GRU(self.decoder_units, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(num_actions, activation='linear')

        self.attention = BahdanauAttention(self.decoder_units)

    def call(self, x, hidden, encoder_output):
        context_vector, attention_weights = self.attention(hidden, encoder_output)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        x = self.fc(output)

        return x, state, attention_weights