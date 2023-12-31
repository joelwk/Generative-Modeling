import re
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read('./GenerativeModeling/generative_text/configcustom.ini')
config_params = config['params']
params = {key: config_params[key] for key in config_params}
config_paths = config['paths']
paths = {key: config_paths[key] for key in config_paths}
base_directory = paths['metadata_path']

class DirectoryManager:
    @staticmethod
    def generate_config(base_directory):
        return {'meta_data_dir': 'meta_data'}

    @staticmethod
    def create_directories(config):
        for key, path in config.items():
            if not path.endswith(('.parquet', '.yaml')):
                os.makedirs(path, exist_ok=True)

class Vocabulary:
    def __init__(self, name):
        PAD_token = 0   # Used for padding short sentences
        SOS_token = 1   # Start-of-sentence token
        EOS_token = 2   # End-of-sentence token
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Includes PAD, SOS, EOS
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            self.longest_sentence = sentence_len
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word.get(index, None)

    def to_index(self, word):
        return self.word2index.get(word, None)

class DataPreparation:
    @staticmethod
    def prep_data(data):
        data['text_pairs'] = data['comment'] + ' , ' + data['response_comment']
        return data[['text_pairs']]


    @staticmethod
    def normalize(line, voc_comment, voc_response_comment):
        if pd.isna(line):
            print(f"Warning: Encountered nan value.")
            return ""
        line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
        line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
        parts = line.split(" , ")
        if len(parts) < 2:
            print(f"Skipping line: {line}")
            return None
        thread, comment = parts[0], parts[1]
        comment = f"{voc_comment.to_index('[start]')} " + comment + f" {voc_comment.to_index('[end]')}"
        return thread, comment

    @staticmethod
    def prepare_and_load_data(replies, voc_comment, voc_response_comment):
        prepared_data = DataPreparation.prep_data(replies)
        text_list = prepared_data['text_pairs'].tolist()
        text_pairs = [DataPreparation.normalize(line, voc_comment, voc_response_comment) for line in text_list if not pd.isna(line)]
        text_pairs = [pair for pair in text_pairs if pair is not None]
        return text_pairs

def initialize_and_prepare(base_directory, replies):
    config = DirectoryManager.generate_config(base_directory)
    DirectoryManager.create_directories(config)
    
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

    text_pairs = DataPreparation.prepare_and_load_data(replies, voc_comment, voc_response_comment)
    
    return text_pairs, voc_comment, voc_response_comment