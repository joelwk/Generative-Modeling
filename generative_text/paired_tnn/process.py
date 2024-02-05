from generative_text.general_tnn.utils.fnProcessing import read_config, pad_punctuation, normalize_text, remove_whitespace, string_to_bool
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

config_path='./generative_text/configpaired.ini'
config_params = read_config(section='params',config_path=config_path)
config_config_params = read_config(section='process-config',config_path=config_path)
params = {key: config_params[key] for key in config_params}

def process_paired_data(data, comment_input='text',comment_target='text'):
    data[comment_input] = data[comment_input].fillna("[MISSING]").astype(str)
    data[comment_target] = data[comment_target].fillna("[MISSING]").astype(str)
    all_texts = data[comment_input].tolist() + data[comment_target].tolist()
    all_texts += ['[start]', '[sep]', '[end]']
    tokenizer = Tokenizer(oov_token="[UNK]")
    tokenizer.fit_on_texts(all_texts)
    comment_seqs = tokenizer.texts_to_sequences(data[comment_input])
    response_seqs = tokenizer.texts_to_sequences(data[comment_target])
    for token in ['[start]', '[sep]', '[end]']:
        if token not in tokenizer.word_index:
            tokenizer.word_index[token] = len(tokenizer.word_index) + 1
    input_seqs = []
    target_seqs = []
    start_token_id = tokenizer.word_index['[start]']
    sep_token_id = tokenizer.word_index['[sep]']
    end_token_id = tokenizer.word_index['[end]']
    for comment_seq, response_seq in zip(comment_seqs, response_seqs):
        input_seq = [start_token_id] + comment_seq + [sep_token_id] + response_seq + [end_token_id]
        target_seq = response_seq + [end_token_id]
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    input_seqs_padded = pad_sequences(input_seqs,  maxlen=int(config_params['max_len']), padding='post', truncating='post').astype('int32') 
    target_seqs_padded = pad_sequences(target_seqs,  maxlen=int(config_params['max_len']), padding='post', truncating='post').astype('int32') 
    return input_seqs_padded, target_seqs_padded, tokenizer.word_index, tokenizer

if __name__ == "__main__":
   input_seqs_padded, target_seqs_padded, vocab, tokenizer = process_paired_data(data, comment_input='text', comment_target='text')