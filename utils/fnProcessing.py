import re
import configparser
import warnings
import string
from urllib.parse import urlparse
from bs4 import BeautifulSoup,MarkupResemblesLocatorWarning
from unicodedata import normalize
import pandas as pd
import numpy as np
from profanity import profanity
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

contraction_mapping = pd.read_json('utils/contraction_mapping.json', typ='series').to_dict()

## Function to pad punctuation in a given string
def pad_punctuation(s):
    s = remove_urls(s)
    s = remove_whitespace(s)
    s = re.sub(f"([{string.punctuation}])", r"\1", s)
    s = re.sub(" +", " ", s)
    return s.strip()

# Function to remove URLs from a text string
def remove_urls(text):
    if isinstance(text, str):
        urls = re.findall(r'http\S+|www.\S+', text)
        for url in urls:
            base = urlparse(url).netloc
            base = re.sub(r'^www\.', '', base)
            text = text.replace(url, base)
        text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        soup = BeautifulSoup(text, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        text = re.sub(r'>>\d+', ' ', text)
        text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
        text = re.sub(r'[^a-zA-Z0-9.,!?\' ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return text

## Function to remove excessive white spaces
def remove_whitespace(text):
    if isinstance(text, str):
        return " ".join(text.split())
    elif text is None:
        return None
    else:
        return text 

## Function to remove profanity from a given text
def remove_profanity(text):
  words = text.split()
  cleaned_words = [("*" * len(word)) if profanity.contains_profanity(word) else word for word in words]
  return " ".join(cleaned_words)

## Combined text cleaning function
def clean_text(text):
  text = remove_urls(text)
  text = remove_profanity(text)
  text = remove_whitespace(text)
  return text

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

    # Dictionary to store original comments by thread ID
    original_comments = {}
    original_date_times = {}

    # Iterate through the replies dataframe
    for index, row in replies.iterrows():
        from_thread = int(row['from'])
        to_thread = int(row['to'])
        # Find the original comment for this thread if it hasn't been found yet
        if from_thread not in original_comments:
            original_post = original_data[original_data['thread_id'] == from_thread]
            if not original_post.empty:
                original_comments[from_thread] = original_post.iloc[0]['text_clean']
                original_date_times[from_thread] = original_post.iloc[0]['posted_date_time']
                replies.at[index, 'is_original'] = True

        replies.at[index, 'comment'] = original_comments.get(from_thread)
        replies.at[index, 'comment_posted_date_time'] = original_date_times.get(from_thread)

        # Find the response comment
        response_post = original_data[original_data['thread_id'] == to_thread]
        if not response_post.empty:
            replies.at[index, 'response_comment'] = response_post.iloc[0]['text_clean']
            replies.at[index, 'response_posted_date_time'] = response_post.iloc[0]['posted_date_time']
    return replies
    
def view_shapes(data):
    for inputs, targets in data.take(1):
        print("Inputs:", inputs)
        print("Targets:", targets)