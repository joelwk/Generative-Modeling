import pickle 
import os
import tensorflow as tf
import os
import configparser

def self_attention(prefix="att", mask=None, key_dim=None, num_heads=None, dropout=0.1, **kwargs):
    if key_dim is None or num_heads is None:
        raise ValueError("key_dim and num_heads must be provided") 
    """Self-attention layers at transformer encoder and decoder. Assumes its
    input is the output from positional encoding layer.
    Args:
        prefix (str): The prefix added to the layer names
        masked (bool): whether to use causal mask. Should be False on encoder and
                       True on decoder. When True, a mask will be applied such that
                       each location only has access to the locations before it.
    """
    inputs = tf.keras.layers.Input(shape=(None, key_dim), dtype='float32', name=f"{prefix}_inputs")
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name=f"{prefix}_attn1")
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm1")
    add = tf.keras.layers.Add(name=f"{prefix}_add1")
    attout = attention(query=inputs, value=inputs, key=inputs, use_causal_mask=mask)
    outputs = norm(add([inputs, attout]))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_att")
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
    
def feed_forward(model_dim, ff_dim, dropout=0.1, prefix="ff"):
    inputs = tf.keras.layers.Input(shape=(None, model_dim), dtype='float32', name=f"{prefix}_inputs")
    """
    Feed-forward layers at transformer encoder and decoder. Assumes its
    input is the output from an attention layer with add & norm, the output
    is the output of one encoder or decoder block
    Args:
        model_dim (int): Output dimension of the feed-forward layer, which
                         is also the output dimension of the encoder/decoder
                         block
        ff_dim (int): Internal dimension of the feed-forward layer
        dropout (float): Dropout rate
        prefix (str): The prefix added to the layer names
        """
    dense1 = tf.keras.layers.Dense(ff_dim, activation='relu', name=f"{prefix}_ff1")
    dense2 = tf.keras.layers.Dense(model_dim, name=f"{prefix}_ff2")
    drop = tf.keras.layers.Dropout(dropout, name=f"{prefix}_drop")
    add = tf.keras.layers.Add(name=f"{prefix}_add3")
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm3")
    ffout = dense1(inputs)
    ffout = drop(ffout)
    ffout = dense2(ffout)
    ffout = drop(ffout)
    outputs = norm(add([inputs, ffout]))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}_ff")
    return model
                              
def encoder(key_dim, ff_dim, num_heads, dropout=0.1, prefix="encoder"):
    inputs = tf.keras.layers.Input(shape=(None, key_dim), name=f"{prefix}_inputs")
    self_att = self_attention(prefix=f"{prefix}_self_att", key_dim=key_dim, num_heads=num_heads, mask=False, dropout=dropout)
    x = self_att(inputs) 
    ff = feed_forward(model_dim=key_dim, ff_dim=ff_dim, dropout=dropout, prefix=f"{prefix}_ff")
    x = ff(x)  
    return tf.keras.Model(inputs=inputs, outputs=x, name=f"{prefix}_model")

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


