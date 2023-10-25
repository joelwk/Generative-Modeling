import os
import configparser
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
import traceback
from tensorflow.keras.models import load_model 
from generative.general_generative.tnn import TransformerBlock, TokenAndPositionEmbedding, causal_attention_mask

config = configparser.ConfigParser()
config.read('./generative/config.ini')
config_params = config['params']
params = {key: config_params[key] for key in config_params}

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    "Custom learning rate for Adam optimizer"
    def __init__(self, key_dim=int(params['key_dim']), warmup_steps=int(params['warmup_steps'])):
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

    lr = CustomSchedule(int(params['key_dim']))
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    inputs = layers.Input(shape=(None,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(int(params['max_len']), int(params['vocab_size']), int(params['embedding_dim']))(inputs)
    attention_scores_list = []
    for _ in range(3): 
        x, attention_scores = TransformerBlock(int(params['n_heads']), int(params['key_dim']), int(params['embedding_dim']), int(params['feed_forward_dim']))(x)
        attention_scores_list.append(attention_scores)
    
    outputs = layers.Dense(int(params['vocab_size']), activation="softmax")(x)
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
    def __init__(self, index_to_word, top_k=15):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temperature):
        if temperature <= 0:
            raise ValueError("Temperature should be greater than 0.")
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs
    
    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            outputs = self.model.predict(x, verbose=0)
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
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        return info

    def reshape_attention(self, att_list):
        reshaped_atts = []
        for att in att_list:
            reshaped_atts.append(att[0, :, -1, :])
        return reshaped_atts
    def on_epoch_end(self, epoch, logs=None):
        self.generate("This year has been", max_tokens=100, temperature=1)