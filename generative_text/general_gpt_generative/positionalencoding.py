import tensorflow as tf
import os
import numpy as np
import configparser

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        position_embeddings = self.get_position_embeddings(length)
        embedded_tokens = self.token_embeddings(inputs)
        return embedded_tokens + position_embeddings

    def get_position_embeddings(self, length):
        position_i = tf.range(self.embed_dim // 2, dtype=tf.float32)
        position_j = tf.pow(10000.0, 2 * (position_i / self.embed_dim))
        position_ij = position_i / position_j
        position_ij = tf.cast(position_ij, tf.float32)

        # Concatenate sin and cos embeddings
        sinusoidal_embedding = tf.stack([tf.sin(position_ij), tf.cos(position_ij)], axis=1)
        sinusoidal_embedding = tf.reshape(sinusoidal_embedding, [1, -1])
        sinusoidal_embedding = tf.pad(sinusoidal_embedding, [[0, 0], [0, self.embed_dim % 2]])  # Pad if embed_dim is odd
        sinusoidal_embedding = tf.reshape(sinusoidal_embedding, [1, self.embed_dim])

        # Ensure the positional embeddings cover the sequence length and are broadcastable over the batch size
        position_embeddings = tf.tile(sinusoidal_embedding, [length, 1])
        position_embeddings = tf.reshape(position_embeddings, [length, self.embed_dim])
        position_embeddings = tf.expand_dims(position_embeddings, 0)  # Add batch dimension

        return position_embeddings

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'vocab_size': self.token_embeddings.input_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

