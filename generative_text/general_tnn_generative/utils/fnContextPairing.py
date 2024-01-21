import pandas as pd
import spacy
import os
import re
import json
import lxml.etree as letree
import mwparserfromhell
import configparser
from urllib.parse import urlparse
import string
from unicodedata import normalize
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from multiprocessing import Pool
from generative_text.general_tnn_generative.utils.fnProcessing import read_config, remove_whitespace, normalize_text
from functools import partial

nlp = spacy.load("en_core_web_sm")

class ContextPairing:
    def __init__(self, data, path_to_dump_file, included_entity_labels, config_path='./generative_text/configkeras.ini'):
        self.nlp = spacy.load("en_core_web_sm")
        self.path_to_dump_file = path_to_dump_file
        self.included_entity_labels = included_entity_labels
        self.found_titles = set()
        self.data = data
        self.config_params = read_config(section='process-config', config_path=config_path)

    def chunkify(self, chunk_size=10000):
        context = letree.iterparse(self.path_to_dump_file, events=('end',), tag='{http://www.mediawiki.org/xml/export-0.10/}page')
        current_chunk = []
        for event, elem in context:
            title = elem.findtext('.//{http://www.mediawiki.org/xml/export-0.10/}title')
            text = elem.findtext('.//{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text')
            current_chunk.append((title, text))
            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []
            elem.clear()
        if current_chunk:
            yield current_chunk

    def extract_entities_and_sentences(self, data):
        def clean_text(text):
            text = str(text)
            return re.sub(r'\s+', ' ', text).strip()

        def extraction(row):
            thread_id = row['thread_id']
            text = clean_text(row['text'])
            posted_date = row['posted_date_time']
            doc = nlp(text)
            result = []
            for sentence in doc.sents:
                sentence_info = {
                    'sentence': sentence.text,
                    'entities': [ent.text for ent in sentence.ents],
                    'entity_labels': [ent.label_ for ent in sentence.ents],
                    'posted_date_time': posted_date,
                    'thread_id': thread_id
                }
                result.append(sentence_info)
            return result
        rows = data.apply(extraction, axis=1)
        return [record for sublist in rows for record in sublist]

    def filter_entities_and_create_dataframe(self, entity_tagged_sentences):
        return pd.DataFrame([info for info in entity_tagged_sentences if any(label in self.included_entity_labels for label in info['entity_labels'])])

    def extract_keywords_from_entities(self, filtered_df):
        keywords = {}
        for _, row in filtered_df.iterrows():
            for entity in row['entities']:
                keywords[entity.lower()] = row['thread_id']
        return keywords

    def process_page(self, title, text):
        wikicode = mwparserfromhell.parse(text)
        return {"title": title, "text": wikicode.strip_code().strip()}

    def process_article_chunk(self, chunk, keywords):
        global found_titles
        relevant_articles = []
        for title, text in chunk:
            title_lower = title.lower()
            if title_lower in keywords and title_lower not in self.found_titles:
                thread_id = keywords[title_lower]
                article = self.process_page(title, text)
                article['thread_id'] = thread_id
                relevant_articles.append(article)
                self.found_titles.add(title_lower)
        return relevant_articles

    def process_chunks(self, context_chunks, keywords):
        with Pool(processes=4) as pool:
            process_chunk_func = partial(self.process_article_chunk, keywords=keywords)
            for chunk in context_chunks:
                if self.found_titles == keywords.keys():
                    break
                results = pool.apply_async(process_chunk_func, args=(chunk,))
                yield from results.get()

    def save_context(self, id_int, topic, entity_tagged_sentences, filtered_df, keywords, relevant_articles_df, context_data):
        dir_loc = os.path.join(self.config_params['dir_loc'], topic)
        if not os.path.exists(dir_loc):
            os.makedirs(dir_loc)
        with open(os.path.join(dir_loc, f'labels_{id_int}_{topic}.txt'), 'w') as file:
            for label in self.included_entity_labels:
                file.write(f"{label}\n")
        with open(os.path.join(dir_loc, f'entity_tagged_sentences_{topic}_{id_int}.json'), 'w') as file:
            json.dump(entity_tagged_sentences, file)
        filtered_df.to_csv(os.path.join(dir_loc, f'matched_{id_int}_{topic}.csv'), index=False)
        keywords_df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'thread_id'])
        keywords_df.to_csv(os.path.join(dir_loc, f'keywords_{id_int}_{topic}.csv'), index=False)
        relevant_articles_df.to_csv(os.path.join(dir_loc, f'articles_{id_int}_{topic}.csv'), index=False)
        context_data.to_csv(os.path.join(dir_loc, f'context_{topic}_{id_int}.csv'), index=False)
        print(f'Saved {topic} data to {dir_loc}')
        
    def save_data(self):
        if self.data is not None:
          dir_loc = os.path.join(self.config_params['dir_loc'], self.config_params['topic'])
          file_name = f"data_{self.config_params['topic']}_{self.config_params['topic_id']}.csv"
          file_path = os.path.join(dir_loc, self.config_params['dir_loc'], file_name)
          self.data.to_csv(file_path, index=False)
          print(f"Data saved to {file_path}")
        else:
            print("No data to save.")

    def run(self):
        entity_tagged_sentences = self.extract_entities_and_sentences(self.data.drop_duplicates(subset='thread_id'))
        filtered_df = self.filter_entities_and_create_dataframe(entity_tagged_sentences)
        keywords = self.extract_keywords_from_entities(filtered_df)
        context_chunks = self.chunkify()
        relevant_articles = list(self.process_chunks(context_chunks, keywords))
        relevant_articles_df = pd.DataFrame(relevant_articles)
        data_test_train = relevant_articles_df[relevant_articles_df['text'] != ''].dropna(subset=['text']).drop_duplicates(subset=['text'])
        data_test_train["text_clean"] = data_test_train["text"].apply(remove_whitespace).apply(normalize_text).astype(str)
        context_data = pd.concat([filtered_df.rename(columns={'sentence':'text'}), data_test_train[~data_test_train['text'].str.contains('REDIRECT', na=False)]])[['text','thread_id']]
        self.save_context(self.config_params['topic_id'], self.config_params['topic'], entity_tagged_sentences, filtered_df, keywords, relevant_articles_df, context_data)
        self.save_data()