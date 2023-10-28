import numpy as np
import pandas as pd
import configparser
import os
import tensorflow as tf
from tensorflow.keras import callbacks, models
from generative.custom_generative.tnn import TransformerBlock, TokenAndPositionEmbedding

config = configparser.ConfigParser()
config.read('./generative/config.ini')
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

class TextGenerator(callbacks.Callback):
    def __init__(self, model, vocab, top_k=15, generation_type='general', beam_width=3, sampling_type='top_k', top_p=0.9):
        self.model = model
        self.vocab = vocab
        self.word_to_index = {word: index for index, word in enumerate(vocab)}
        self.top_k = top_k
        self.generation_type = generation_type
        self.beam_width = beam_width
        self.sampling_type = sampling_type
        self.top_p = top_p
        self.info = []

    def sample_from(self, probs, temperature=1.0):
        scaled_probs = probs ** (1 / temperature)
        scaled_probs /= np.sum(scaled_probs)
        return np.random.choice(len(scaled_probs), p=scaled_probs), scaled_probs

    def sample_token(self, logits, temperature=1.0):
        probs = tf.nn.softmax(logits).numpy()
        if self.sampling_type == 'top_k':
            top_k_indices = tf.math.top_k(logits, k=self.top_k).indices.numpy()
            logits = logits[top_k_indices]
            probs = tf.nn.softmax(logits).numpy()
            sample_token, _ = self.sample_from(probs, temperature)
            return top_k_indices[sample_token], probs
        elif self.sampling_type == 'top_p':
            sorted_indices = np.argsort(logits)[::-1]
            sorted_probs = tf.nn.softmax(logits[sorted_indices]).numpy()
            cumulative_probs = np.cumsum(sorted_probs)
            filtered_indices = sorted_indices[cumulative_probs <= self.top_p]
            sample_token, _ = self.sample_from(sorted_probs, temperature)
            return filtered_indices[sample_token], sorted_probs

    def append_info(self, prompt, logits, probs):
        self.info.append({
            "prompt": prompt,
            "logits": logits,
            "probs": probs
        })

    def generate(self, start_prompt, max_tokens, temperature=1.0):
        start_tokens = [self.word_to_index.get(word, 1) for word in start_prompt.split()]

        if self.generation_type == 'beam':
            sequences = [[list(start_tokens), 0.0]]
            for _ in range(max_tokens):
                all_candidates = []
                for seq in sequences:
                    x = np.array([seq[0]])
                    y = self.model.predict(x, verbose=0)[0]
                    logits = y[0, :]
                    probs = tf.nn.softmax(logits).numpy()
                    top_indices = tf.math.top_k(logits, k=self.top_k).indices.numpy()
                    for i in top_indices:
                        candidate = [seq[0] + [i], seq[1] - np.log(probs[i])]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                sequences = ordered[:self.beam_width]
            generated_tokens = sequences[0][0]
            self.append_info(start_prompt, logits, probs)
            
        elif self.generation_type == 'greedy':
            generated_tokens = []
            for _ in range(max_tokens):
                x = np.array([start_tokens])
                y = self.model.predict(x, verbose=0)[0]
                logits = y[0, -1]
                probs = tf.nn.softmax(logits).numpy()
                sample_token = np.argmax(logits)
                generated_tokens.append(sample_token)
                start_tokens.append(sample_token)
                if sample_token == 0:
                    break
            self.append_info(start_prompt, logits, probs)
            
        else:  # general
            for _ in range(max_tokens):
                x = np.array([start_tokens])
                logits = self.model.predict(x)[0][-1]
                sample_token, probs = self.sample_token(logits, temperature)
                self.append_info(start_prompt, logits, probs)
                start_tokens.append(sample_token)
                
        return " ".join([self.vocab[token] for token in start_tokens])

    def on_test_end(self, logs=None):
        generated_text = self.generate("Draft", max_tokens=50)
        print(f"Generated text: {generated_text}")
        print(f"Info: {self.info[0]}")