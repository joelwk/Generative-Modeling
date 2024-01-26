import os
import configparser
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
import traceback
from tensorflow.keras.models import load_model 
from generative_text.general_tnn_generative.tnn import TransformerBlock, TokenAndPositionEmbedding, causal_attention_mask
from generative_text.general_tnn_generative.utils.fnProcessing import read_config,pad_punctuation, normalize_text, remove_whitespace

config_params = read_config(section='params', config_path='./generative_text/configkeras.ini')
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

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    "Custom learning rate for Adam optimizer"
    def __init__(self, key_dim=key_dim, warmup_steps=warmup_steps):
        super().__init__()
        self.key_dim = key_dim
        self.warmup_steps = warmup_steps
        self.d = tf.cast(self.key_dim, tf.float32)
    @tf.function
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

def train_model(preload_model=False, model_path=None):

    lr = CustomSchedule(key_dim, warmup_steps)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=epsilon)
    inputs = layers.Input(shape=(None,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim)(inputs)
    attention_scores_list = []
    for _ in range(num_layers): 
        x, attention_scores = TransformerBlock(num_heads, key_dim, embedding_dim, ff_dim, dropout_rate)(x)
        attention_scores_list.append(attention_scores)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    gpt = models.Model(inputs=inputs, outputs=[outputs] + attention_scores_list)
    gpt.compile(optimizer=optimizer, loss=[losses.SparseCategoricalCrossentropy()] + [None]*5)
    if preload_model and model_path:
        if os.path.exists(model_path):
            try:
                print("Loading pre-existing model.")
                gpt = load_model(model_path, custom_objects={
                    "TransformerBlock": TransformerBlock,
                    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                    "CustomSchedule": CustomSchedule
                })
            except Exception as e:
                print(f"Failed to load model. Error: {e}")
                traceback.print_exc() 
        else:
            print(f"Model path {model_path} does not exist.")
    return gpt

class TrainTextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, tokenizer, top_k=15):
        self.index_to_word = index_to_word
        self.tokenizer = tokenizer
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}

    def sample_from(self, probs, temperature):
        if temperature <= 0:
            raise ValueError("Temperature should be greater than 0.")
        probs = np.ones(len(self.index_to_word)) * probs
        probs = probs / np.sum(probs)
        return np.random.choice(len(self.index_to_word), p=probs), probs
        
    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = self.tokenizer([start_prompt])
        start_tokens = np.array(start_tokens[0])
        start_tokens = pad_sequences([start_tokens], maxlen=max_len, padding='post')
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array(start_tokens).reshape(1, -1)
            outputs = self.model.predict(x, verbose=1)
            y = outputs[0] 
            att = outputs[1:]
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            reshaped_atts = self.reshape_attention(att)
            info.append(
                {
                    "prompt": start_prompt,
                    "word_probs": probs,
                    "atts": reshaped_atts,
                }
            )
            start_tokens = np.append(start_tokens, sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        return info

    def reshape_attention(self, att_list):
        reshaped_atts = []
        for att in att_list:
            reshaped_atts.append(att[0, :, -1, :])
        return reshaped_atts

    def on_epoch_end(self, epoch, logs=None):
        self.generate("This year has been", max_tokens=max_len, temperature=1)