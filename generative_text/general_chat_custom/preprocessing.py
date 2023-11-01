import re
import pandas as pd
import os
import configparser
config = configparser.ConfigParser()
config.read('./generative_text/configcustom.ini')
config_params = config['params']
params = {key: config_params[key] for key in config_params}
base_directory = params['dataset_path']

class DirectoryManager:
    @staticmethod
    def generate_config(base_directory):
        return {
            'meta_data_dir': 'meta_data',
        }

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
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # Longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]
class DataPreparation:
    @staticmethod
    def prep_data(data):
        data['text_pairs'] = data['comment'] +' , '+ data['response_comment']
        return data[['text_pairs']]
    @staticmethod
    def normalize(line):
        if pd.isna(line):
            print(f"Warning: Encountered nan value.")
            return ""
        line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
        line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
        line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
        line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
        line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
        parts = line.split(" , ")
        
        if len(parts) < 2:
            print(f"Skipping line: {line}")
            return None
        thread, comment = parts[0],parts[1]
        comment = "[start] " + comment + " [end]"
        return thread, comment
    @staticmethod
    def prepare_and_load_data(replies):
        prepared_text = DataPreparation.prep_data(replies)
        text_list = [i for i in prepared_text.text_pairs]

        text_pairs = [DataPreparation.normalize(line) for line in text_list if not pd.isna(line)]

        corpus_comments = replies['comment'].tolist()
        voc_comment = Vocabulary('comments')
        for sent in corpus_comments:
            if not pd.isna(sent):
                voc_comment.add_sentence(sent)

        corpus_threads = replies['response_comment'].tolist()
        voc_response_comment = Vocabulary('voc_response_comment')
        for sent in corpus_threads:
            if not pd.isna(sent):
                voc_response_comment.add_sentence(sent)

        return text_pairs, voc_comment, voc_response_comment

def initialize_and_prepare(base_directory, replies):
    config = DirectoryManager.generate_config(base_directory)
    DirectoryManager.create_directories(config)
    return DataPreparation.prepare_and_load_data(replies)