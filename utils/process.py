import configparser
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import os
import pandas as pd
import logging 
from utils.fnProcessing import pad_punctuation, remove_urls, remove_whitespace, clean_text

config = configparser.ConfigParser()
config.read('./generative/config.ini')
params = config["params"]

def prepare_data(data):
    data["text_clean"] = data["posted_comment"].apply(remove_urls).apply(remove_whitespace).apply(pad_punctuation).astype(str)
    return data[data['text_clean'].notnull()]

def get_datasets(text_data):
    text_data = tf.data.Dataset.from_tensor_slices(text_data)
    n_samples = sum(1 for _ in text_data)
    
    # Split into training, validation, and test sets
    n_train_samples = int(n_samples * 0.7)
    n_val_samples = int(n_samples * 0.2)
    
    train_text_ds = text_data.take(n_train_samples)
    val_text_ds = text_data.skip(n_train_samples).take(n_val_samples)
    test_text_ds = text_data.skip(n_train_samples + n_val_samples)

    # Shuffle and batch
    batch_size = config.getint("params", "batch_size")
    train_text_ds = train_text_ds.batch(batch_size).shuffle(buffer_size=n_train_samples)
    val_text_ds = val_text_ds.batch(batch_size).shuffle(buffer_size=n_val_samples)
    test_text_ds = test_text_ds.batch(batch_size).shuffle(buffer_size=n_samples - n_train_samples - n_val_samples)

    return train_text_ds, val_text_ds, test_text_ds
    
def main(data):
    clean_data = prepare_data(data)
    text_data = [text[:config.getint("params", "max_len")] for text in clean_data['text_clean'].tolist()]
    text_data = [text.lower() for text in text_data]
    
    # Adapt the TextVectorization layer
    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=config.getint("params", "vocab_size"),
        output_mode="int",
        output_sequence_length=config.getint("params", "max_len") + 1
    )
    
    train_text_ds, val_text_ds, test_text_ds = get_datasets(text_data)
    vectorize_layer.adapt(train_text_ds)
    
    def prepare_lm_inputs_labels(text):
        """
        Shift word sequences by 1 position so that the target for position (i) is
        word at position (i+1). The model will use all words up till position (i)
        to predict the next word.
        """
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y

    train_ds = train_text_ds.map(prepare_lm_inputs_labels)
    val_ds = val_text_ds.map(prepare_lm_inputs_labels)
    test_ds = test_text_ds.map(prepare_lm_inputs_labels)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    # Get the vocabulary
    vocab = vectorize_layer.get_vocabulary()
    return train_ds, val_ds, test_ds, vocab
    
if __name__ == "__main__":
    main(data)