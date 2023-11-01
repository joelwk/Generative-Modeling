import tensorflow as tf
import os
import numpy as np
import configparser
config = configparser.ConfigParser()
config.read('./generative_text/configcustom.ini')
config_params = config['params']
params = {key: config_params[key] for key in config_params}
max_len = int(params['max_len'])
vocab_size = int(params['vocab_size'])
embedding_dim = int(params['embedding_dim'])
num_heads = int(params['n_heads'])
num_layers = int(params['n_layers'])
key_dim = int(params['key_dim'])
ff_dim = int(params['feed_forward_dim'])
dropout_rate = float(params['dropout'])
warmup_steps = int(params['warmup_steps'])
activation = params['activation']
epsilon = 1e-6 
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_seq_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = self.build_position_embeddings()

    def build_position_embeddings(self):
        position_embeddings = np.zeros((self.max_seq_length, self.embed_dim))
        for pos in range(self.max_seq_length):
            for i in range(0, self.embed_dim, 2):
                position_embeddings[pos, i] = np.sin(pos / 10000 ** (i / self.embed_dim))
                position_embeddings[pos, i + 1] = np.cos(pos / 10000 ** ((i + 1) / self.embed_dim))
        return tf.constant(position_embeddings[None, :, :], dtype=tf.float32)

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        position_embeddings_slice = self.position_embeddings[:, :sequence_length, :]
        return embedded_tokens + position_embeddings_slice

    def compute_mask(self, *args, **kwargs):
        return super().compute_mask(*args, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_seq_length': self.max_seq_length,
            'embed_dim': self.embed_dim,
            'vocab_size': self.token_embeddings.input_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract vocab_size from config
        vocab_size = config.pop('vocab_size')
        # Initialize an instance of PositionalEmbedding
        positional_embedding_instance = cls(vocab_size=vocab_size, **config)
        # Reconstruct the Embedding layer
        positional_embedding_instance.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=config['embed_dim'])
        return positional_embedding_instance