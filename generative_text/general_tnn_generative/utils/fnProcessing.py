import re
import os
import configparser
import warnings
import string
from urllib.parse import urlparse
from bs4 import BeautifulSoup,MarkupResemblesLocatorWarning
from bs4 import BeautifulSoup,MarkupResemblesLocatorWarning
from unicodedata import normalize
import pandas as pd
import numpy as np
from profanity import profanity

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
config_path='./generative_text/configkeras.ini'

def read_config(section="params", config_path='./../'):
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config.sections():
        print(f"Section '{section}' not found in configuration.")
        raise KeyError(f"Section not found: {section}")
    return {key: config[section][key] for key in config[section]}

url_regex = re.compile(r'http\S+|www.\S+')
whitespace_regex = re.compile(r'\s+')
punctuation_regex = re.compile(f"([{string.punctuation}])")
non_alphanumeric_regex = re.compile(r'[^a-zA-Z0-9.,!?\' ]')
punctuation_regex = re.compile(f"([{string.punctuation}])")
contraction_mapping = pd.read_json('./generative_text/general_tnn_generative/utils/contraction_mapping.json', typ='series').to_dict()
config = read_config(section="params", config_path=config_path)
config_params = read_config(section="process-config", config_path=config_path)
   
wiki_markup_regex = re.compile(
    r'thumb\|[\dpx]*\|?|'
    r'right\|[\dpx]*\|?|'
    r'left\|[\dpx]*\|?|'
    r'center\|[\dpx]*\|?|'
    r'[\dpx]+\|'
)

def pad_punctuation(s):
    if string_to_bool(config_params.get("padding", "False")):
        if not isinstance(s, str):
            return ""
        s = punctuation_regex.sub(r" \1 ", s)
        print(s)
        return whitespace_regex.sub(' ', s).strip()
    return s
    
def normalize_text(text):
    if isinstance(text, str):
        try:
            # Existing normalization steps
            text = url_regex.sub(lambda m: urlparse(m.group(0)).netloc.replace('www.', ''), text)
            text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            text = wiki_markup_regex.sub('', text)  # Remove wiki markup
            text = re.sub(r'\n\n.*?\n\n*?\n', ' ', text)
            text = text.replace('\n', ' ')
            text = ' '.join(BeautifulSoup(text, 'html.parser').stripped_strings)
            text = re.sub(r'>>\d+', ' ', text)
            # Revised pattern to remove 'thumb|', 'px', '200px|', 'right|', and similar patterns
            text = re.sub(r'thumb\|\d*x\d*px\|right\|', '', text)
            text = re.sub(r'thumb\|\d*x\d*px\|', '', text)
            text = re.sub(r'thumb\|', '', text)
            text = re.sub(r'\d*x\d*px\|', '', text)
            # Existing normalization steps continued
            if string_to_bool(config_params.get("contraction_mapping", "False")):
                text = ' '.join(contraction_mapping.get(t, t) for t in text.split())
            if string_to_bool(config_params.get("non_alpha_numeric", "False")):
                text = non_alphanumeric_regex.sub(' ', text)

            return whitespace_regex.sub(' ', text).strip()
        except ValueError:
            return text
    return text

def remove_whitespace(text):
    if isinstance(text, str):
        return " ".join(text.split())
    return text

def remove_profanity(text):
  words = text.split()
  cleaned_words = [("*" * len(word)) if profanity.contains_profanity(word) else word for word in words]
  return " ".join(cleaned_words)

def find_dialogs(data):
    pattern = re.compile(r'&gt;&gt;\d+')
    to_from = []
    for index, row in data.iterrows():
        thread_id = int(row['thread_id'])
        posted_comment = row['posted_comment']
        references = pattern.findall(posted_comment)
        for reference in references:
            ref_id = int(reference.replace('&gt;&gt;', ''))
            if ref_id != thread_id:
                to_from.append((ref_id, thread_id))
    dialog_df = pd.DataFrame(to_from, columns=['from', 'to'], dtype='int64')
    return dialog_df
    
def augment_dialogs(replies, original_data):
    replies['is_original'] = False
    replies['comment'] = None
    replies['response_comment'] = None
    replies['comment_posted_date_time'] = None
    replies['response_posted_date_time'] = None
    original_comments = {}
    original_date_times = {}
    for index, row in replies.iterrows():
        from_thread = int(row['from'])
        to_thread = int(row['to'])
        if from_thread not in original_comments:
            original_post = original_data[original_data['thread_id'] == from_thread]
            if not original_post.empty:
                original_comments[from_thread] = original_post.iloc[0]['text_clean']
                original_date_times[from_thread] = original_post.iloc[0]['posted_date_time']
                replies.at[index, 'is_original'] = True
        replies.at[index, 'comment'] = original_comments.get(from_thread)
        replies.at[index, 'comment_posted_date_time'] = original_date_times.get(from_thread)
        response_post = original_data[original_data['thread_id'] == to_thread]
        if not response_post.empty:
            replies.at[index, 'response_comment'] = response_post.iloc[0]['text_clean']
            replies.at[index, 'response_posted_date_time'] = response_post.iloc[0]['posted_date_time']
    return replies
    
def view_shapes(data):
    for inputs, targets in data.take(1):
        print("Inputs:", inputs)
        print("Targets:", targets)

def load_csvs(directory_path, starts_with):
    dataframes = []
    for file in os.listdir(directory_path):
        if file.startswith(str(starts_with)):
            file_path = os.path.join(directory_path, file)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    data = pd.concat(dataframes, ignore_index=True)
    return data

def string_to_bool(string_value):
    return string_value.lower() in ['true', '1', 't', 'y', 'yes', 'on']