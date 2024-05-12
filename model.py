import tensorflow as tf


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(
        q, k, transpose_b=True
    )  # (..., variates_num_q, variates_num_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., variates_num_q, variates_num_k)

    output = tf.matmul(attention_weights, v)  # (..., variates_num_q, depth_v)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, variates_num, d_model)
        k = self.wk(k)  # (batch_size, variates_num, d_model)
        v = self.wv(v)  # (batch_size, variates_num, d_model)

        q = self.split_heads(
            q, batch_size
        )  # (batch_size, num_heads, variates_num_q, depth)
        k = self.split_heads(
            k, batch_size
        )  # (batch_size, num_heads, variates_num_k, depth)
        v = self.split_heads(
            v, batch_size
        )  # (batch_size, num_heads, variates_num_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, variates_num_q, depth)
        scaled_attention = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, variates_num_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, variates_num_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, variates_num_q, d_model)

        return output


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                dff, activation="gelu"
            ),  # (batch_size, variates_num, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, variates_num, d_model)
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)  # (batch_size, variates_num, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, variates_num, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, variates_num, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, variates_num, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        rate=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

    def call(self, x, training):
        x = self.embedding(x)  # (batch_size, variates_num, d_model)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)

        return x  # (batch_size, variates_num, d_model)


class ITransformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        tar_seq_len,
        rate=0.1,
        name="i_transformer",
    ):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)

        self.final_layer = tf.keras.layers.Dense(tar_seq_len)

    def call(self, inp, training):
        inp = tf.transpose(
            inp, perm=[0, 2, 1]
        )  # (batch_size, variates_num, inp_seq_len)

        enc_output = self.encoder(
            inp, training=training
        )  # (batch_size, variates_num, d_model)

        final_output = self.final_layer(
            enc_output
        )  # (batch_size, variates_num, tar_seq_len)

        final_output = tf.transpose(
            final_output, perm=[0, 2, 1]
        )  # (batch_size, tar_seq_len, variates_num)

        return final_output
