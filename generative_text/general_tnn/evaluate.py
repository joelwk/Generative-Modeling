import numpy as np
import pandas as pd
import configparser
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.preprocessing.sequence import pad_sequences
from generative_text.general_tnn.tnn import TransformerBlock, TokenAndPositionEmbedding
from utils.fnProcessing import read_config

config_params = read_config(section='params-general', config_path='./generative_text/config.ini')
params = {key: config_params[key] for key in config_params}

max_len = int(params['max_len'])
vocab_size = int(params['vocab_size'])
num_heads = int(params['n_heads'])
num_layers = int(params['n_layers'])
key_dim = int(params['key_dim'])
embedding_dim = key_dim * num_heads
ff_dim = embedding_dim * int(params['ff_multiplier'])
dropout_rate = float(params['dropout'])
warmup_steps = int(params['warmup_steps'])
activation = params['activation']
epsilon = float(params['epsilon'])

class TextGenerator:
    def __init__(self, model, max_len, vocab, vectorize_layer, top_k=40, top_p=0.9, sampling_type='random', temperature=1.0):
        self.model = model
        self.max_len = max_len
        self.vocab = vocab
        self.vectorize_layer = vectorize_layer
        self.index_to_word = {i: word for i, word in enumerate(vocab)}
        self.top_k = top_k
        self.top_p = top_p
        self.sampling_type = sampling_type
        self.temperature = temperature

    def sample_from(self, probs, temperature):
        scaled_probs = np.exp(np.log(probs) / temperature)
        scaled_probs /= np.sum(scaled_probs)
        return np.random.choice(len(scaled_probs), p=scaled_probs), scaled_probs

    def sample_token(self, logits, temperature):
        probs = tf.nn.softmax(logits).numpy()
        if self.sampling_type == 'top_k':
            top_k_indices = tf.math.top_k(logits, k=self.top_k).indices.numpy()
            top_k_logits = logits[top_k_indices]
            top_k_probs = tf.nn.softmax(top_k_logits).numpy()
            sample_index = np.random.choice(range(self.top_k), p=top_k_probs)
            return top_k_indices[sample_index]
        elif self.sampling_type == 'top_p':
            sorted_indices = np.argsort(logits)[::-1]
            sorted_probs = tf.nn.softmax(logits[sorted_indices]).numpy()
            cumulative_probs = np.cumsum(sorted_probs)
            indices_to_remove = cumulative_probs > self.top_p
            sorted_probs[indices_to_remove] = 0
            sorted_probs /= np.sum(sorted_probs)
            sample_index = np.random.choice(range(len(sorted_probs)), p=sorted_probs)
            return sorted_indices[sample_index]
        else:
            return self.sample_from(probs, temperature)[0]

    def _generate_text_base(self, user_input, sampling_func):
        if user_input:
            start_tokens = self.vectorize_layer([user_input])[0].numpy()
            token_list = list(start_tokens)
        else:
            start_token = np.random.choice(len(self.vocab))
            token_list = [start_token]
        num_tokens_to_generate = self.max_len - len(token_list)
        for _ in range(num_tokens_to_generate):
            input_padded = self._prepare_input(token_list)
            output_seq = self.model.predict(input_padded, verbose=0)
            current_token_index = len(token_list) - 1
            logits = output_seq[0][0, current_token_index, :]
            next_token = sampling_func(logits, self.temperature)
            token_list.append(next_token)
            if next_token == 0:
                break
        generated_text = " ".join([self.index_to_word.get(token, '?') for token in token_list])
        return generated_text.strip()

    def random_sampling(self, logits, temperature):
        sample_index, _ = self.sample_from(tf.nn.softmax(logits).numpy(), temperature)
        return sample_index

    def greedy_search(self, logits, _):
        return np.argmax(logits)

    def _prepare_input(self, token_list):
        input_seq = tf.expand_dims(token_list, axis=0)
        return pad_sequences(input_seq, maxlen=self.max_len, padding='post')

    def beam_search(self, user_input, beam_size=3):
        if user_input:
            start_tokens = self.vectorize_layer([user_input])[0].numpy()
        else:
            random_word = self.index_to_word[np.random.choice(len(self.vocab))]
            start_tokens = self.vectorize_layer([random_word])[0].numpy()
        beams = [(0, start_tokens.tolist())]
        for _ in range(len(start_tokens), self.max_len):
            new_beams = []
            for prob, token_seq in beams:
                input_padded = self._prepare_input(token_seq)
                prediction = self.model.predict(input_padded, verbose=0)[0]
                last_step_prediction = prediction[0, len(token_seq)-1, :]
                top_probs, top_indices = tf.math.top_k(tf.nn.softmax(last_step_prediction), k=beam_size)
                for i in range(beam_size):
                    next_token_prob = top_probs[i].numpy()
                    next_token_index = top_indices[i].numpy()
                    new_prob = prob + np.log(next_token_prob)
                    new_token_seq = token_seq + [next_token_index]
                    new_beams.append((new_prob, new_token_seq))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        best_beam = max(beams, key=lambda x: x[0])
        generated_sequence = best_beam[1]
        generated_text = " ".join([self.index_to_word.get(token, '?') for token in generated_sequence])
        return generated_text.strip()

    def generate_text(self, user_input=None, method='random'):
        if method == 'random' or method == 'greedy':
            sampling_func = self.sample_token
            return self._generate_text_base(user_input, sampling_func)
        elif method == 'beam':
            return self.beam_search(user_input, beam_size=3)
        else:
            raise ValueError("Invalid generation method specified.")