import configparser
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import os
import pandas as pd
import logging 
from utils.fnProcessing import pad_punctuation, remove_urls, remove_whitespace, clean_text
from generative.tnn import TransformerBlock, TokenAndPositionEmbedding  
from generative.train import train_model, TrainTextGenerator, CustomSchedule, TrainTextGenerator

config = configparser.ConfigParser()
config.read('./generative/config.ini')
params = config["params"]

def main(data):
    data["text_clean"] = data["posted_comment"].apply(remove_urls).apply(remove_whitespace).apply(pad_punctuation).astype(str)
    data = data[data['text_clean'].notnull()]
    text_data = [text[:int(params['max_len'])] for text in data['text_clean'].tolist()]

    text_data = [text.lower() for text in text_data]
    text_data = tf.data.Dataset.from_tensor_slices(text_data)

    # Adapt a temporary TextVectorization layer to your data
    temp_vectorize_layer = TextVectorization(standardize=None)
    temp_vectorize_layer.adapt(text_data)
    dataset_vocab = set(temp_vectorize_layer.get_vocabulary())

    # Remove reserved tokens from dataset vocab
    dataset_vocab.discard("")
    dataset_vocab.discard("[UNK]")

    # Combine GloVe and dataset vocabularies (no duplicates)
    combined_vocab = ["", "[UNK]"] + list(dataset_vocab)[:int(params['vocab_size']) - 2]

    # Create the TextVectorization layer
    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=int(params['vocab_size']),
        output_mode="int",
        output_sequence_length=int(params['max_len']) + 1,
    )

    # Set the vocabulary
    vectorize_layer.set_vocabulary(combined_vocab)

    # Calculate the number of samples
    n_samples = len(text_data)

    # Calculate the number of samples in each set
    n_train_samples = int(n_samples * 0.70)  # 70% of data for training
    n_val_samples = int(n_samples * 0.20)  # 20% of data for validation
    n_test_samples = n_samples - n_train_samples - n_val_samples  # 10% of data for testing

    # Split the dataset into train, validation, and test sets
    train_text_ds = text_data.take(n_train_samples)
    val_text_ds = text_data.skip(n_train_samples).take(n_val_samples)
    test_text_ds = text_data.skip(n_train_samples + n_val_samples)

    # Shuffle and batch each set individually
    train_text_ds = train_text_ds.shuffle(buffer_size=n_train_samples).batch(int(params['batch_size']))
    val_text_ds = val_text_ds.shuffle(buffer_size=n_val_samples).batch(int(params['batch_size']))
    test_text_ds = test_text_ds.shuffle(buffer_size=n_test_samples).batch(int(params['batch_size']))

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

    # Prepare the datasets for language model training
    train_ds = train_text_ds.map(prepare_lm_inputs_labels)
    val_ds = val_text_ds.map(prepare_lm_inputs_labels)
    test_ds = test_text_ds.map(prepare_lm_inputs_labels)

    # Prefetch the datasets
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, combined_vocab
    
if __name__ == "__main__":
    data = pd.read_parquet('./data.parquet').sample(5000)  # Replace with different ranges or filtered data - see notebook for more detail
    main(data)