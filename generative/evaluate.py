import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import numpy as np
from generative.tnn import TransformerBlock, TokenAndPositionEmbedding
import pandas as pd
import configparser
import os
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, key_dim, warmup_steps):
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
        return {"key_dim": self.key_dim, "warmup_steps": self.warmup_steps}

class TextGenerator:
    def __init__(self, model, index_to_word, top_k=5):
        self.model = model
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}

    def reshape_attention(self, att_list):
        return [att[0, :, -1, :] for att in att_list]

    def sample_from(self, probs, temperature):
        probs = (probs ** (1 / temperature)) / np.sum(probs ** (1 / temperature))
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
        info = []
        while len(start_tokens) < max_tokens:
            x = np.array([start_tokens])
            outputs = self.model.predict(x)
            y_probs = outputs[0][0][-1]  # Assuming probabilities are at this location
            sample_token, probs = self.sample_from(y_probs, temperature)
            info.append({
                "prompt": start_prompt,
                "atts": self.reshape_attention(outputs[1:]),
                "temperature": temperature,
                "word_probs": probs
            })
            if sample_token == 0: break
            start_tokens.append(sample_token)
            start_prompt += " " + self.index_to_word[sample_token]
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info