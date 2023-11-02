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
import sentencepiece as spm
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
# Ensure you have the necessary NLTK data
# nltk.download('punkt')

def process_and_load_data(replies):
    # Constants
    config = configparser.ConfigParser()
    config.read('./generative_text/configcustom.ini')
    config_params = config['params']
    params = {key: config_params[key] for key in config_params}
    config_paths = config['paths']
    paths = {key: config_paths[key] for key in config_paths}
    base_directory = paths['metadata_path']
    validation_split = float(params['validation_split'])
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
    random_seed = int(params['random_seed'])
    buffer_size= int(params['buffer_size'])
    batch_size= int(params['batch_size'])
    pre_batch_size= int(params['pre_batch_size'])
    epsilon = 1e-6 
    config_directories = DirectoryManager.generate_config(base_directory)
    DirectoryManager.create_directories(config_directories) 
    meta_data_dir = config_directories['meta_data_dir']

    text_pairs, voc_comment, voc_response_comment = initialize_and_prepare(base_directory, replies)

    def format_dataset(comment, comment_target, comment_vectorizer, target_comment_vectorizer):
        comment = comment_vectorizer(comment)
        comment_target = target_comment_vectorizer(comment_target)
        # No padding is done here; handle dynamically within the model or during batching
        encoder_input = comment
        decoder_input = tf.concat([[0], comment_target[:-1]], axis=0)  # Assuming 0 is the start token
        target = comment_target
        return ({"encoder_inputs": encoder_input, "decoder_inputs": decoder_input}, target)

    def make_dataset(pairs, comment_vectorizer, target_comment_vectorizer):
        thread_texts, comment_texts = zip(*pairs)
        dataset = tf.data.Dataset.from_tensor_slices((list(thread_texts), list(comment_texts)))
        dataset = dataset.shuffle(buffer_size, seed=random_seed) \
                        .map(lambda x, y: format_dataset(x, y, comment_vectorizer, target_comment_vectorizer)) \
                        .padded_batch(batch_size, padded_shapes=({'encoder_inputs': [None], 'decoder_inputs': [None]}, [None])) \
                        .prefetch(pre_batch_size)
        return dataset

    def prepare_data():
        input_texts, target_texts = zip(*text_pairs)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
        tokenizer.fit_on_texts(input_texts + target_texts)
        vocab_size = len(tokenizer.word_index) + 1
        # Shuffle and split
        random.seed(random_seed)
        random.shuffle(text_pairs)
        total_samples = len(text_pairs)
        n_val = int(validation_split * len(text_pairs))
        n_train = len(text_pairs) - 2 * n_val
        train_pairs = text_pairs[:n_train]
        val_pairs = text_pairs[n_train:n_train + n_val]
        test_pairs = text_pairs[n_train + n_val:]
        
        # Vectorization
        train_comment_texts = [pair[0] for pair in train_pairs]
        train_response_comment_texts = [pair[1] for pair in train_pairs]
        input_tokens = set(token for text in input_texts for token in text.split())
        target_tokens = set(token for text in target_texts for token in text.split())
        comment_tokens, response_comment_tokens = set(), set()
        comment_maxlen, response_maxlen = 0, 0
        for comment, response in text_pairs:
            comment_tok, response_tok = comment.split(), response.split()
            comment_maxlen = max(comment_maxlen, len(comment_tok))
            response_maxlen = max(response_maxlen, len(response_tok))
            comment_tokens.update(comment_tok)
            response_comment_tokens.update(response_tok)
            
        comment_vectorizer = TextVectorization(
            standardize=None,
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=max_len
        )
        
        response_comment_vectorizer = TextVectorization(
            standardize=None,
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=max_len + 1
        )

        comment_vectorizer.adapt(train_comment_texts)
        response_comment_vectorizer.adapt(train_response_comment_texts)

        # Serialize the model configurations and weights
        with open(f'{meta_data_dir}/vectorize.pickle', "wb") as fp:
            data_to_pickle = {
                "train": train_pairs,
                "val": val_pairs,
                "test": test_pairs,
                "commentvec_config": comment_vectorizer.get_config(),
                "commentvec_weights": comment_vectorizer.get_weights(),
                "commentresponsevec_config": response_comment_vectorizer.get_config(),
                "commentresponsevec_weights": response_comment_vectorizer.get_weights(),
            }
            pickle.dump(data_to_pickle, fp)

        # Save the vocabulary
        comment_vocab = comment_vectorizer.get_vocabulary()
        response_comment_vocab = response_comment_vectorizer.get_vocabulary()
        
        # Serialize vocabulary
        with open(f'{meta_data_dir}/comment_vocab.pickle', 'wb') as f:
            pickle.dump(comment_vocab, f)
        with open(f'{meta_data_dir}/response_comment_vocab.pickle', 'wb') as f:
            pickle.dump(response_comment_vocab, f)
        return comment_vectorizer, response_comment_vectorizer, train_pairs, val_pairs, test_pairs
        
    if not os.path.exists(os.path.join(meta_data_dir, f"vectorize.pickle")):
        comment_vectorizer, response_comment_vectorizer, train_pairs, val_pairs, test_pairs = prepare_data()
        # Serialize the model configurations and weights
        with open(f'{meta_data_dir}/vectorize.pickle', "wb") as fp:
            data_to_pickle = {
                "train": train_pairs,
                "val": val_pairs,
                "test": test_pairs,
                "commentvec_config": comment_vectorizer.get_config(),
                "commentvec_weights": comment_vectorizer.get_weights(),
                "commentresponsevec_config": response_comment_vectorizer.get_config(),
                "commentresponsevec_weights": response_comment_vectorizer.get_weights(),}
            pickle.dump(data_to_pickle, fp)
    else:
        with open(os.path.join(meta_data_dir, f"vectorize.pickle"), "rb") as fp:
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