
from generative_text.general_tnn_generative.utils.fnProcessing import read_config, pad_punctuation, normalize_text, remove_whitespace, string_to_bool

config_path='./generative_text/configkeras.ini'
config_params = read_config(section='params',config_path=config_path)
config_config_params = read_config(section='process-config',config_path=config_path)
params = {key: config_params[key] for key in config_params}

import tensorflow as tf
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

batch_size = int(params['batch_size'])
validation_split = float(params['validation_split'])
base_max_len = int(params['max_len'])
vocab_size = int(params['vocab_size'])

def prepare_data(data, input_col, clean_col):
    data[input_col] = data[input_col].astype(str)
    data[clean_col] = data[input_col].apply(normalize_text).apply(remove_whitespace).apply(pad_punctuation)
    return data[data[clean_col].notnull() & data[clean_col].str.strip().astype(bool)]

def dynamic_pad_sequences(batch, padding_value=0):
    max_len = max([len(x) for x in batch])
    padded_batch = [x + [padding_value] * (max_len - len(x)) for x in batch]
    return np.array(padded_batch)

def get_datasets(text_data):
    test_split = float(validation_split / 2)
    if not isinstance(text_data, tf.data.Dataset):
        text_data = tf.data.Dataset.from_tensor_slices(text_data)
    n_samples = sum(1 for _ in text_data)
    n_train_samples = int(n_samples * (1 - validation_split - test_split))
    n_val_samples = int(n_samples * validation_split)
    n_test_samples = n_samples - n_train_samples - n_val_samples
    assert n_train_samples > 0, "Training set has no samples."
    assert n_val_samples > 0, "Validation set has no samples."
    assert n_test_samples > 0, "Test set has no samples."
    text_data = text_data.shuffle(buffer_size=n_samples)
    train_text_ds = text_data.take(n_train_samples).batch(batch_size)
    val_text_ds = text_data.skip(n_train_samples).take(n_val_samples).batch(batch_size)
    test_text_ds = text_data.skip(n_train_samples + n_val_samples).take(n_test_samples).batch(batch_size)
    return train_text_ds, val_text_ds, test_text_ds

def main(data, input_col='text', clean_col='text'):
    if data is None or data.empty:
        raise ValueError("Data is None or empty, cannot proceed with processing.")
    if string_to_bool(config_config_params.get("process_data", "False")):
        data = prepare_data(data, input_col, clean_col)
    if data is not None:
        text_data = [text.lower() for text in data[clean_col].tolist()]
        vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=None,
            max_tokens=vocab_size,
            output_mode="int")
        train_text_ds, val_text_ds, test_text_ds = get_datasets(text_data)
        vectorize_layer.adapt(train_text_ds)

    def prepare_lm_inputs_labels(text):
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y
        
    train_ds = train_text_ds.map(prepare_lm_inputs_labels)
    val_ds = val_text_ds.map(prepare_lm_inputs_labels)
    test_ds = test_text_ds.map(prepare_lm_inputs_labels)
    vocab = vectorize_layer.get_vocabulary()
    return train_ds, val_ds, test_ds, vocab, vectorize_layer

if __name__ == "__main__":
    train_ds, val_ds, test_ds, vocab, vectorize_layer = main(data, input_col='text', clean_col='text')