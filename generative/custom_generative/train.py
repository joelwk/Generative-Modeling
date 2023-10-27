import os
import configparser
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
import traceback
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model 
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

def train_model(preload_model=False, model_path=None, use_sinusoidal=False, return_attention_scores=False, num_layers=num_layers):
    lr = CustomSchedule(key_dim)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    inputs = layers.Input(shape=(None,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim, use_sinusoidal=use_sinusoidal, initializer="glorot_uniform")(inputs)
    
    attention_scores_list = []
    
    for _ in range(num_layers):  # Loop through the number of layers
        x, attention_scores = TransformerBlock(
            num_heads, 
            key_dim, 
            embedding_dim, 
            ff_dim, 
            dropout_rate=dropout_rate, 
            epsilon=float(epsilon), 
            activation=activation,
            num_layers=num_layers
        )(x, return_attention_scores=True)
        
        if return_attention_scores:
            attention_scores_list.append(attention_scores)
            
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    gpt = models.Model(inputs=inputs, outputs=[outputs] + (attention_scores_list if return_attention_scores else []))
    
    gpt.compile(optimizer=optimizer, loss=[losses.SparseCategoricalCrossentropy()] + ([None]*len(attention_scores_list) if return_attention_scores else []))
    
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
    def __init__(self, combined_vocab, top_k=15):
        super().__init__()
        self.index_to_word = combined_vocab
        self.word_to_index = {word: index for index, word in enumerate(combined_vocab)}
    def set_model(self, model):
        self.model = model

    def sample_from(self, probs, temperature):
        if np.isscalar(probs):
            print("Warning: Received a scalar. Expected a probability distribution.")
            return int(probs), probs
        elif len(probs.shape) == 1 and np.isclose(np.sum(probs), 1.0):
            scaled_probs = probs ** (1 / temperature)
            scaled_probs /= np.sum(scaled_probs)
            return np.random.choice(len(scaled_probs), p=scaled_probs), scaled_probs
        else:
            raise ValueError("Invalid input: Expected a probability distribution.")
                
    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [self.word_to_index.get(word, 1) for word in start_prompt.split()]
        info = []
        while len(start_tokens) < max_tokens:
            x = np.array([start_tokens])
            outputs = self.model.predict(x, verbose=0)
            y = outputs[0]
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            start_tokens.append(sample_token)
            start_prompt += " " + (self.index_to_word[sample_token] if sample_token < len(self.index_to_word) else "")
            print("Shape of y:", y.shape)
            print("Shape of y[0]:", y[0].shape)

            info.append({
                "prompt": start_prompt,
                "word_probs": probs
            })
            if sample_token == 0:
                break
        return info
        
    def reshape_attention(self, att_list):
        return [att[0, :, -1, :] for att in att_list]

    def on_epoch_end(self, epoch, logs=None):
        self.generate("This year has been", max_tokens=100, temperature=1)

