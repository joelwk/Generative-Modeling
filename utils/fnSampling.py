import pandas as pd
import random
import glob
import os
import logging
from datetime import datetime
from utils.fnProcessing import remove_urls, remove_whitespace

logging.basicConfig(level=logging.INFO)

def stratified_sample(data, strata_column, sample_ratio):
    sample_size = lambda x: max(int(len(x) * sample_ratio), 1)
    return data.groupby(strata_column).apply(lambda x: x.sample(n=sample_size(x)))

def reservoir_sampling(iterator, k):
    reservoir = []
    for i, item in enumerate(iterator):
        if i < k: reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k: reservoir[j] = item
    return reservoir

def str_to_date(date_str, format="%m%y"):
    return datetime.strptime(date_str, format)

def within_date_range(file_prefix, start_date, end_date):
    file_date = str_to_date(file_prefix)
    return start_date <= file_date <= end_date

def stratified_sample_by_time(data, time_column, freq, sample_ratio, datetime_format=None):
    data['temp_time_column'] = pd.to_datetime(data[time_column], format=datetime_format)
    sampled_data = data.groupby(pd.Grouper(key='temp_time_column', freq=freq)).apply(lambda x: x.sample(frac=sample_ratio) if len(x) > 0 else x)
    sampled_data.reset_index(drop=True, inplace=True)
    del sampled_data['temp_time_column']
    return sampled_data

def sample_data_by_prefix(directory, time_column, freq, sample_ratio, start_date=None, end_date=None):
    datetime_format = "%Y-%m-%d %H:%M:%S" 
    sampled_data = pd.DataFrame()
    for file_name in os.listdir(directory):
        if not file_name.endswith('.parquet'):
            continue
        file_prefix = file_name.split('_')[0]
        
        if start_date and end_date:
            if not within_date_range(file_prefix, str_to_date(start_date), str_to_date(end_date)):
                continue

        file_path = os.path.join(directory, file_name)
        data = pd.read_parquet(file_path)
        
        stratified_data = stratified_sample_by_time(data, time_column, freq, sample_ratio, datetime_format)        
        sampled_data = pd.concat([sampled_data, stratified_data], ignore_index=True)
        
        logging.info(f"Processed file {file_name}. Sampled Data Size: {len(sampled_data)}")
    return sampled_data

def count_total_rows(directory):
    total_rows = 0
    files_count = 0
    for file_name in [f for f in os.listdir(directory) if f.endswith('.parquet')]:
        total_rows += len(pd.read_parquet(os.path.join(directory, file_name)))
        files_count += 1
    print(f'Total number of files processed {files_count} containing: {total_rows} rows')
    return total_rows