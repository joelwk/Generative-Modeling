import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Embedding, Input
from tensorflow.keras.models import Model
from generative_text.paired_tnn.tnn import MultiHeadAttention, TransformerBlock, CustomSchedule, build_transformer_model
from generative_text.general_tnn.utils.fnProcessing import read_config, pad_punctuation, normalize_text, remove_whitespace
from generative_text.paired_tnn.process import process_paired_data
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import callbacks

config_params = read_config(section='params', config_path='./generative_text/configpaired.ini')
params = {key: config_params[key] for key in config_params}

max_len = int(params['max_len'])
vocab_size = int(params['vocab_size'])
num_heads = int(params['n_heads'])
num_layers = int(params['n_layers'])
embed_dim = int(params['embed_dim'])
feed_forward = int(embed_dim * int(params['ff_multiplier']))
key_dim = int(embed_dim//2)
dropout_rate = float(params['dropout'])
warmup_steps = int(params['warmup_steps'])
activation = params['activation']
epsilon = float(params['epsilon'])
epochs = int(params['epochs'])
batch_size = int(params['batch_size'])
validation_size = float(params['validation_split'])

class PairedTNNTextGenerator(callbacks.Callback):
    def __init__(self, tokenizer, max_len=50, top_k=15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.top_k = top_k

    def sample_with_temperature(self, logits, temperature=1.0):
        """Apply temperature scaling and sample an index."""
        probabilities = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return np.random.choice(len(probabilities), p=probabilities)

    def generate_response(self, seed_text, max_len=50, temperature=1.0):
        start_token_str = '[start]'
        sep_token_str = '[sep]'
        end_token_str = '[end]'
        formatted_seed_text = f"{start_token_str} {seed_text} {sep_token_str}"
        input_text = formatted_seed_text
        for _ in range(max_len):
            sequence = self.tokenizer.texts_to_sequences([input_text])[0]
            padded_sequence = pad_sequences([sequence], maxlen=self.max_len, padding='post')
            prediction = self.model.predict(padded_sequence, verbose=0)
            predicted_token_id = self.sample_with_temperature(prediction[0, -1, :], temperature)
            predicted_word = self.tokenizer.index_word.get(predicted_token_id, '')
            if predicted_word == end_token_str or predicted_word == '':
                break
            input_text += ' ' + predicted_word
        generated_text = input_text.split(sep_token_str)[-1].strip()
        return generated_text.replace(end_token_str, '').strip()

    def on_epoch_end(self, epoch, logs=None):
        seed_text = "i am happy because"
        generated_response = self.generate_response(seed_text, max_len=self.max_len, temperature=0.7)
        print(f"\nSample generated response for '{seed_text}': {generated_response}")

def prepare_model_training(data, preload_model=False, model_path=None):
    train_ds, val_ds, test_ds, vocab, tokenizer = process_paired_data(data, comment_input='comment', comment_target='response_comment')
    vocab_size = len(vocab) + 1
    lr_schedule = CustomSchedule(embed_dim=embed_dim, warmup_steps=warmup_steps)
    model = build_transformer_model(vocab_size, embed_dim, num_heads, feed_forward, num_layers, dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=epsilon)
    model.compile(optimizer=optimizer, loss=[losses.SparseCategoricalCrossentropy()])
    if preload_model and model_path:
        if os.path.exists(model_path):
            try:
                print("Loading pre-existing model.")
                model = load_model(model_path, custom_objects={
                    "TransformerBlock": TransformerBlock,
                    "CustomSchedule": CustomSchedule
                })
            except Exception as e:
                print(f"Failed to load model. Error: {e}")
                traceback.print_exc() 
        else:
            print(f"Model path {model_path} does not exist.")
    return model

if __name__ == "__main__":
    train_ds, val_ds, test_ds, vocab, tokenizer, model = prepare_model_training()