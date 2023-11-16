import pickle 
import os
import tensorflow as tf
import os
import configparser

def self_attention(prefix="att", key_dim=None, num_heads=None, dropout=0.1, **kwargs):
    if key_dim is None or num_heads is None:
        raise ValueError("key_dim and num_heads must be provided")
    inputs = tf.keras.layers.Input(shape=(None, key_dim), dtype='float32', name=f"{prefix}_inputs")
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name=f"{prefix}_attn")
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm")
    add = tf.keras.layers.Add(name=f"{prefix}_add")
    seq_len = tf.shape(inputs)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    attout = attention(query=inputs, value=inputs, key=inputs, attention_mask=look_ahead_mask)
    outputs = norm(add([inputs, attout]))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_att")
    return model

def feed_forward(model_dim, ff_dim, dropout=0.1, prefix="ff"):
    inputs = tf.keras.layers.Input(shape=(None, model_dim), dtype='float32', name=f"{prefix}_inputs")
    dense1 = tf.keras.layers.Dense(ff_dim, activation='relu', name=f"{prefix}_ff1")
    dense2 = tf.keras.layers.Dense(model_dim, name=f"{prefix}_ff2")
    drop = tf.keras.layers.Dropout(dropout, name=f"{prefix}_drop")
    add = tf.keras.layers.Add(name=f"{prefix}_add")
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm")

    ffout = dense1(inputs)
    ffout = drop(ffout)
    ffout = dense2(ffout)
    ffout = drop(ffout)
    outputs = norm(add([inputs, ffout]))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_ff")
    return model

def cross_attention(prefix="att", key_dim=None, num_heads=None, dropout=0.1):
    if key_dim is None or num_heads is None:
        raise ValueError("key_dim and num_heads must be provided")
    """Cross-attention layers at transformer decoder. Assumes its
    input is the output from positional encoding layer at decoder
    and context is the final output from encoder.
    Args:
        prefix (str): The prefix added to the layer names
    """
    context = tf.keras.layers.Input(shape=(None, key_dim), dtype='float32', name=f"{prefix}_context")
    inputs = tf.keras.layers.Input(shape=(None, key_dim), dtype='float32', name=f"{prefix}_inputs")
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name=f"{prefix}_attn2")
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm2")
    add = tf.keras.layers.Add(name=f"{prefix}_add2")
    attout = attention(query=inputs, value=context, key=context)
    outputs = norm(add([inputs, attout]))
    model = tf.keras.Model(inputs=[inputs, context], outputs=outputs, name=f"{prefix}_cross")
    return model

def decoder(key_dim, ff_dim, num_heads, dropout=0.1, prefix="decoder"):
    inputs = tf.keras.layers.Input(shape=(None, key_dim), name=f"{prefix}_inputs")
    context = tf.keras.layers.Input(shape=(None, key_dim), name=f"{prefix}_context")
    self_att = self_attention(prefix=f"{prefix}_self_att", key_dim=key_dim, num_heads=num_heads, mask=True, dropout=dropout)
    x = self_att(inputs)
    cross_att = cross_attention(prefix=f"{prefix}_cross_att", key_dim=key_dim, num_heads=num_heads, dropout=dropout)
    x = cross_att([x, context])
    ff = feed_forward(model_dim=key_dim, ff_dim=ff_dim, dropout=dropout, prefix=f"{prefix}_ff")
    x = ff(x)
    return tf.keras.Model(inputs=[inputs, context], outputs=x, name=f"{prefix}_model")


