import numpy as np
import pandas as pd
import configparser
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from generative.general_generative.tnn import TransformerBlock, TokenAndPositionEmbedding

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
    def __init__(self, model, index_to_word, top_k=5, generation_type='general', beam_width=3, sampling_type='top_k', top_p=0.9):
        self.model = model
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}
        self.top_k = top_k
        self.generation_type = generation_type
        self.beam_width = beam_width
        self.sampling_type = sampling_type
        self.top_p = top_p

    def sample_from(self, probs, temperature):
        scaled_probs = probs ** (1 / temperature)
        scaled_probs /= np.sum(scaled_probs)
        return np.random.choice(len(scaled_probs), p=scaled_probs), scaled_probs

    def sample_token(self, logits, temperature):
        probs = tf.nn.softmax(logits).numpy()
        if self.sampling_type == 'top_k':
            top_k_indices = tf.math.top_k(logits, k=self.top_k).indices.numpy()
            logits = logits[top_k_indices]
            probs = tf.nn.softmax(logits).numpy()
            sample_token, _ = self.sample_from(probs, temperature)
            return top_k_indices[sample_token]
        elif self.sampling_type == 'top_p':
            sorted_indices = np.argsort(logits)[::-1]
            sorted_probs = tf.nn.softmax(logits[sorted_indices]).numpy()
            cumulative_probs = np.cumsum(sorted_probs)
            filtered_indices = sorted_indices[cumulative_probs <= self.top_p]
            sample_token, _ = self.sample_from(sorted_probs, temperature)
            return filtered_indices[sample_token]

    def reshape_attention(self, att_list):
        return [att[0, :, -1, :] for att in att_list]

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
        info = []
        
        def append_info(prompt, outputs, temperature, probs):
            info.append({
                "prompt": prompt,
                "atts": self.reshape_attention(outputs[1:]),
                "temperature": temperature,
                "word_probs": probs
            })
        # Greedy generation
        if self.generation_type == 'greedy':
            while len(start_tokens) < max_tokens:
                x = np.array([start_tokens])
                outputs = self.model.predict(x)
                y_probs = outputs[0][0][-1]
                sample_token = np.argmax(y_probs)
                append_info(start_prompt, outputs, temperature, y_probs)
                if sample_token == 0: break
                start_tokens.append(sample_token)
                start_prompt += " " + self.index_to_word[sample_token]
        # Beam search generation
        elif self.generation_type == 'beam':
            sequences = [[list(start_tokens), 0.0]]
            for _ in range(max_tokens):
                all_candidates = []
                for seq in sequences:
                    x = np.array([seq[0]])
                    outputs = self.model.predict(x)
                    y_probs = outputs[0][0][-1]
                    probs = tf.nn.softmax(y_probs).numpy()
                    top_indices = tf.math.top_k(y_probs, k=self.top_k).indices.numpy()
                    for i in top_indices:
                        candidate = [seq[0] + [i], seq[1] - np.log(probs[i])]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                sequences = ordered[:self.beam_width]
            start_tokens = sequences[0][0]
            start_prompt = " ".join([self.index_to_word[token] for token in start_tokens])
            append_info(start_prompt, outputs, temperature, y_probs)
        # General generation
        else:
            while len(start_tokens) < max_tokens:
                x = np.array([start_tokens])
                outputs = self.model.predict(x)
                y_probs = outputs[0][0][-1]
                sample_token, probs = self.sample_from(y_probs, temperature)
                append_info(start_prompt, outputs, temperature, probs)
                if sample_token == 0: break
                start_tokens.append(sample_token)
                start_prompt += " " + self.index_to_word[sample_token]
        
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info