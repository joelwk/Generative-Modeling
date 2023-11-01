import pickle 
import os
import tensorflow as tf
import os
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

from generative_text.general_chat_custom.layers import cross_attention, self_attention, feed_forward, encoder, decoder
from generative_text.general_chat_custom.PositionalEmbedding import PositionalEmbedding

def transformer(num_layers, num_heads, seq_len, key_dim, ff_dim, vocab_size_src,
                vocab_size_tgt, dropout=0.1, name="transformer"):
    embed_shape = (seq_len, key_dim)  # output shape of the positional embedding layer
    # set up layers
    input_enc = tf.keras.layers.Input(shape=(seq_len,), dtype="int32",
                                      name="encoder_inputs")
    input_dec = tf.keras.layers.Input(shape=(seq_len,), dtype="int32",
                                      name="decoder_inputs")
    embed_enc = PositionalEmbedding(seq_len, vocab_size_src, key_dim, name="embed_enc")
    embed_dec = PositionalEmbedding(seq_len, vocab_size_tgt, key_dim, name="embed_dec")

    final = tf.keras.layers.Dense(vocab_size_tgt, name="linear")
    encoder_model = encoder(input_shape=(seq_len, key_dim), key_dim=key_dim, ff_dim=ff_dim,
                            num_heads=num_heads)
    decoder_model = decoder(input_shape=(seq_len, key_dim), key_dim=key_dim, ff_dim=ff_dim,
                            num_heads=num_heads)
    # build output
    x1 = embed_enc(input_enc)
    x2 = embed_dec(input_dec)
    for _ in range(num_layers):
        x1 = encoder_model(x1)
    for _ in range(num_layers):
        x2 = decoder_model([x2, x1])
    output = final(x2)
    # XXX keep this try-except block
    try:
        del output._keras_mask
    except AttributeError:
        pass
    model = tf.keras.Model(inputs=[input_enc, input_dec], outputs=output, name=name)
    return model

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, key_dim, warmup_steps=25000):
        super().__init__()
        self.key_dim = key_dim
        self.warmup_steps = warmup_steps
        self.d = tf.cast(self.key_dim, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            "key_dim": self.key_dim,
            "warmup_steps": self.warmup_steps,
        }
        return config