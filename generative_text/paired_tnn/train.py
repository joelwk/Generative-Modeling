import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Embedding, Input
from tensorflow.keras.models import Model
from generative_text.paired_tnn.tnn import MultiHeadAttention, TransformerBlock, CustomSchedule, build_transformer_model
from generative_text.general_tnn.utils.fnProcessing import read_config, pad_punctuation, normalize_text, remove_whitespace
from generative_text.paired_tnn.process import process_paired_data
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

config_params = read_config(section='params', config_path='./generative_text/configpaired.ini')
params = {key: config_params[key] for key in config_params}

max_len = int(params['max_len'])
vocab_size = int(params['vocab_size'])
num_heads = int(params['n_heads'])
num_layers = int(params['n_layers'])
embed_dim = int(params['embed_dim'])
feed_forward = int(embed_dim * 2)
key_dim = int(embed_dim//2)
dropout_rate = float(params['dropout'])
warmup_steps = int(params['warmup_steps'])
activation = params['activation']
epsilon = float(params['epsilon'])
epochs = int(params['epochs'])
batch_size = int(params['batch_size'])
validation_size = float(params['validation_split'])

def train_model(data, comment_input, comment_target):
    input_seqs_padded, target_seqs_padded, vocab, tokenizer = process_paired_data(data, comment_input=comment_input, comment_target=comment_target)
    vocab_size = len(vocab) + 1
    lr_schedule = CustomSchedule(embed_dim=embed_dim, warmup_steps=warmup_steps)
    model = build_transformer_model(vocab_size, embed_dim, num_heads, feed_forward, num_layers, dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=epsilon)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    input_seqs_padded = input_seqs_padded[:, :-1]
    target_seqs_padded = target_seqs_padded[:, 1:]
    input_train, input_val, target_train, target_val = train_test_split(input_seqs_padded, target_seqs_padded, test_size=validation_size, random_state=42)
    return model.fit(input_train, target_train,
                    validation_data=(input_val, target_val),
                    epochs=epochs,
                    callbacks=[EarlyStopping(patience=25, monitor='val_loss')],
                    batch_size=batch_size), model, vocab, tokenizer

if __name__ == "__main__":
    history, model, vocab, tokenizer = train_model(data, comment_input, comment_target)