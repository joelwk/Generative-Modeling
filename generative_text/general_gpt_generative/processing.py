import os
import re
import random
import pickle
import pandas as pd
import tensorflow as tf
import configparser
import pandas as pd
from utils.fnProcessing import find_dialogs, augment_dialogs,view_shapes
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from GenerativeModeling.generative_text.general_gpt_generative.preprocessing import DirectoryManager, Vocabulary

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

config = configparser.ConfigParser()
config.read('./GenerativeModeling/generative_text/configcustom.ini')
config_params = config['params']
config_paths = config['paths']
paths = {key: config_paths[key] for key in config_paths}
base_directory = paths['metadata_path']
params = {key: config_params[key] for key in config_params}
max_len = int(params['max_len'])
embedding_dim = int(params['embedding_dim'])
num_heads = int(params['n_heads'])
num_layers = int(params['n_layers'])
key_dim = int(params['key_dim'])
ff_dim = int(params['feed_forward_dim'])
dropout_rate = float(params['dropout'])
warmup_steps = int(params['warmup_steps'])
activation = params['activation']
epoch = int(params['epochs'])
validation_split = float(params['validation_split'])
random_seed = int(params['random_seed'])
buffer_size = int(params['buffer_size'])
batch_size = int(params['batch_size'])
pre_batch_size = int(params['pre_batch_size'])
config_directories = DirectoryManager.generate_config(base_directory)
DirectoryManager.create_directories(config_directories)
meta_data_dir = config_directories['meta_data_dir']

def process_data(replies, base_directory, config):
    # Create directories as needed
    DirectoryManager.create_directories(DirectoryManager.generate_config(base_directory))

    # Initialize Vocabulary and prepare text pairs
    voc_comment = Vocabulary('comments')
    voc_response_comment = Vocabulary('response_comments')
    start_token = '[start]'
    end_token = '[end]'
    voc_comment.add_word(start_token)
    voc_comment.add_word(end_token)
    voc_response_comment.add_word(start_token)
    voc_response_comment.add_word(end_token)

    for sent in replies['comment'].tolist():
        if not pd.isna(sent):
            voc_comment.add_sentence(sent)
    for sent in replies['response_comment'].tolist():
        if not pd.isna(sent):
            voc_response_comment.add_sentence(sent)

    text_pairs = [(comment, f"{start_token} {response_comment} {end_token}")
                  for comment, response_comment in zip(replies['comment'], replies['response_comment'])
                  if not pd.isna(comment) and not pd.isna(response_comment)]

    def format_dataset(combined_sequence, vectorizer):
        vectorized_sequence = vectorizer(combined_sequence)
        input_sequence = vectorized_sequence[:-1]
        target_sequence = vectorized_sequence[1:]
        return (input_sequence, target_sequence)

    def make_dataset(pairs, vectorizer):
        combined_texts = [comment + " [start] " + response_comment + " [end]" for comment, response_comment in pairs]
        dataset = tf.data.Dataset.from_tensor_slices(combined_texts)
        dataset = dataset.shuffle(buffer_size, seed=random_seed) \
            .map(lambda x: format_dataset(x, vectorizer), num_parallel_calls=tf.data.AUTOTUNE) \
            .padded_batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)
        return dataset

    def prepare_data():
        input_texts, target_texts = zip(*text_pairs)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
        tokenizer.fit_on_texts(input_texts + target_texts)
        vocab_size = len(tokenizer.word_index) + 1
        random.seed(random_seed)
        random.shuffle(text_pairs)
        total_samples = len(text_pairs)
        n_val = int(validation_split * len(text_pairs))
        n_train = len(text_pairs) - 2 * n_val
        train_pairs = text_pairs[:n_train]
        val_pairs = text_pairs[n_train:n_train + n_val]
        test_pairs = text_pairs[n_train + n_val:]

        vectorizer = TextVectorization(
            standardize=None,
            max_tokens=vocab_size,
            output_mode="int",)

        combined_texts = [comment + " [start] " + response_comment + " [end]" for comment, response_comment in train_pairs]
        vectorizer.adapt(combined_texts)
        return vectorizer, train_pairs, val_pairs, test_pairs

    # Load or prepare vectorizer and dataset
    if not os.path.exists(os.path.join(meta_data_dir, "vectorize.pickle")):
        vectorizer, train_pairs, val_pairs, test_pairs = prepare_data()
    else:
        with open(os.path.join(meta_data_dir, "vectorize.pickle"), "rb") as fp:
            data = pickle.load(fp)
        vectorizer = TextVectorization.from_config(data["vectorizer_config"])
        vectorizer.set_weights(data["vectorizer_weights"])
        train_pairs = data["train"]
        val_pairs = data["val"]
        test_pairs = data["test"]

    train_ds = make_dataset(train_pairs, vectorizer)
    val_ds = make_dataset(val_pairs, vectorizer)
    test_ds = make_dataset(test_pairs, vectorizer)
    return train_ds, val_ds, test_ds, vectorizer,text_pairs