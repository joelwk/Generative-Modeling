import os
import configparser
import tensorflow as tf
from tensorflow.keras import layers

config = configparser.ConfigParser()
config.read('./generative/config.ini')
config_params = config["params"]
params = {key: config_params[key] for key in config_params}

# Read parameters from the config
max_len = int(params['max_len'])
vocab_size = int(params['vocab_size'])
embedding_dim = int(params['embedding_dim'])
num_heads = int(params['n_heads'])
key_dim = int(params['key_dim'])
ff_dim = int(params['feed_forward_dim'])
dropout_rate = float(params['dropout'])
activation = params['activation']
epsilon = 1e-6 

def get_angles(pos, i, d_model):
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        tf.range(position)[:, tf.newaxis],
        tf.range(d_model)[tf.newaxis, :],
        d_model
    )
    angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class FeedForwardNetwork(layers.Layer):
    def __init__(self, ff_dim, activation=activation, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.ffn_1 = layers.Dense(ff_dim, activation=activation)
        self.ffn_2 = layers.Dense(ff_dim)
    def call(self, x):
        return self.ffn_2(self.ffn_1(x))

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate, epsilon, activation, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon 
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(num_heads, key_dim, output_shape=embed_dim)
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=epsilon)
        self.ffn = FeedForwardNetwork(ff_dim, activation)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=epsilon)
        
    @tf.function 
    def call(self, inputs, attention_mask=None, return_attention_scores=False):
        attention_output, attention_scores = self.attn(
            inputs, inputs, attention_mask=attention_mask, return_attention_scores=True
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_2(ffn_output)
        out2 = self.ln_2(out1 + ffn_output)
        
        if return_attention_scores:
            return out2, attention_scores
        else:
            return out2
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'activation': self.activation
        })
        return config
        
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, use_sinusoidal=False, initializer="glorot_uniform", **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_sinusoidal = use_sinusoidal
        self.initializer = initializer
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=initializer)
        if not use_sinusoidal:
            self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim, embeddings_initializer=initializer)

    @tf.function
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        if self.use_sinusoidal:
            positions = positional_encoding(maxlen, self.embed_dim)
        else:
            positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'use_sinusoidal': self.use_sinusoidal,
            'initializer': self.initializer
        })
        return config

token_and_position_embedding = TokenAndPositionEmbedding(
    max_len=max_len,
    vocab_size=vocab_size,
    embed_dim=embedding_dim
)
transformer_block = TransformerBlock(
    num_heads=num_heads,
    key_dim=key_dim,
    embed_dim=embedding_dim,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate,
    epsilon=epsilon,
    activation=activation 
)
feed_forward_network = FeedForwardNetwork(
    ff_dim=ff_dim
)
