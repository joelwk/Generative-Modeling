import random
import re
import os
import pickle
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pandas as pd
import tensorflow as tf
import configparser
from generative_text.general_chat_custom.preprocessing import DirectoryManager  
from generative_text.general_chat_custom.preprocessing import initialize_and_prepare  


def process_and_load_data(replies, save_to_disk=True):
    # Load configuration settings
    config = configparser.ConfigParser()
    config.read('./generative_text/configcustom.ini')
    params = {key: config['params'][key] for key in config['params']}
    paths = {key: config['paths'][key] for key in config['paths']}
    base_directory = paths['metadata_path']
    validation_split = float(params['validation_split'])
    max_len = int(params['max_len'])
    config_vocab_size = int(params['vocab_size']) 
    random_seed = int(params['random_seed'])
    buffer_size = int(params['buffer_size'])
    batch_size = int(params['batch_size'])
    pre_batch_size = int(params['pre_batch_size'])
    config_directories = DirectoryManager.generate_config(base_directory)
    DirectoryManager.create_directories(config_directories)
    meta_data_dir = config_directories['meta_data_dir']

    # Load the data
    text_pairs, voc_comment, voc_response_comment = initialize_and_prepare(base_directory, replies)

    def format_dataset(comment_x, response_comment_y, comment_vectorizer, response_comment_vectorizer):
        encoder_input = comment_vectorizer(comment_x) 
        response_comment_target = response_comment_vectorizer(response_comment_y)
        decoder_input = tf.concat([[1], response_comment_target[:-1]], axis=0)
        target = response_comment_target
       # tf.print("Encoder input shape:", tf.shape(encoder_input))
       # tf.print("Decoder input shape:", tf.shape(decoder_input))
       # tf.print("Target shape:", tf.shape(target))
        return ({"encoder_inputs": encoder_input, "decoder_inputs": decoder_input}, target)

    def make_dataset(pairs, comment_vectorizer, response_comment_vectorizer):
        response_comment_texts, comment_texts = zip(*pairs)
        dataset = tf.data.Dataset.from_tensor_slices((list(comment_texts), list(response_comment_texts)))
        dataset = dataset.shuffle(buffer_size, seed=random_seed) \
            .map(lambda x, y: format_dataset(x, y, comment_vectorizer, response_comment_vectorizer), 
                num_parallel_calls=tf.data.AUTOTUNE) \
                    .padded_batch(batch_size) \
                        .prefetch(tf.data.AUTOTUNE)
       # tf.print("Number of batches in the dataset:", tf.data.experimental.cardinality(dataset))
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
        train_comment_texts = [pair[0] for pair in train_pairs]
        train_response_comment_texts = [pair[1] for pair in train_pairs]
        
        comment_vectorizer = TextVectorization(
            standardize=None,
            max_tokens=len(list(voc_comment.word2index.keys())),
            output_mode="int",)
        response_comment_vectorizer = TextVectorization(
            standardize=None,
            max_tokens=len(list(voc_response_comment.word2index.keys())),
            output_mode="int",)

        comment_vectorizer.adapt(train_comment_texts)
        response_comment_vectorizer.adapt(train_response_comment_texts)
        return comment_vectorizer, response_comment_vectorizer, train_pairs, val_pairs, test_pairs
    if not os.path.exists(os.path.join(meta_data_dir, "vectorize.pickle")):
        comment_vectorizer, response_comment_vectorizer, train_pairs, val_pairs, test_pairs = prepare_data()
    else:
        with open(os.path.join(meta_data_dir, "vectorize.pickle"), "rb") as fp:
            data = pickle.load(fp)
        comment_vectorizer = TextVectorization.from_config(data["commentvec_config"])
        comment_vectorizer.set_weights(data["commentvec_weights"])
        response_comment_vectorizer = TextVectorization.from_config(data["commentresponsevec_config"])
        response_comment_vectorizer.set_weights(data["commentresponsevec_weights"])
        train_pairs = data["train"]
        val_pairs = data["val"]
        test_pairs = data["test"]

    train_ds = make_dataset(train_pairs, comment_vectorizer, response_comment_vectorizer)
    val_ds = make_dataset(val_pairs, comment_vectorizer, response_comment_vectorizer)
    test_ds = make_dataset(test_pairs, comment_vectorizer, response_comment_vectorizer)
    return train_ds, val_ds, test_ds, comment_vectorizer, response_comment_vectorizer