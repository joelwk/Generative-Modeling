import pickle 
import os
import tensorflow as tf
import os
import configparser
from generative_text.general_chat_custom.layers import cross_attention, self_attention, feed_forward, encoder, decoder
from generative_text.general_chat_custom.PositionalEmbedding import PositionalEmbedding

def transformer(num_layers, num_heads, key_dim, ff_dim, vocab_size_src, vocab_size_tgt, dropout, name="transformer"):
    input_enc = tf.keras.layers.Input(shape=(None,), dtype="int32", name="encoder_inputs")
    input_dec = tf.keras.layers.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    embed_enc = PositionalEmbedding(vocab_size_src, key_dim, name="embed_enc")(input_enc)
    embed_dec = PositionalEmbedding(vocab_size_tgt, key_dim, name="embed_dec")(input_dec)
    encoder_layers = [encoder(key_dim=key_dim, ff_dim=ff_dim, num_heads=num_heads, dropout=dropout, prefix=f"encoder_layer_{i}") for i in range(num_layers)]
    decoder_layers = [decoder(key_dim=key_dim, ff_dim=ff_dim, num_heads=num_heads, dropout=dropout, prefix=f"decoder_layer_{i}") for i in range(num_layers)]
    x1 = embed_enc
    x2 = embed_dec
    for i, enc_layer in enumerate(encoder_layers):
        x1 = enc_layer(x1)
    for i, dec_layer in enumerate(decoder_layers):
        x2 = dec_layer([x2, x1])
    final_output = tf.keras.layers.Dense(vocab_size_tgt, name="final_output")(x2)
    model = tf.keras.Model(inputs=[input_enc, input_dec], outputs=final_output, name=name)
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