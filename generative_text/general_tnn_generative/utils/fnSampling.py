import pandas as pd
import random
import glob
import os
import logging
from datetime import datetime
from generative_text.general_tnn_generative.fnProcessing import remove_urls, remove_whitespace

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

def safe_str_to_date(date_str, format="%Y-%m-%d %H:%M:%S"):
    try:
        if isinstance(date_str, datetime):
            return date_str
        return datetime.strptime(date_str, format)
    except (ValueError, TypeError) as e:
        print(f"Error converting '{date_str}' to date format '{format}':", e)
        return None

def within_date_range(date, start_date, end_date):
    if date is None:
        return False
    return (start_date is None or date >= start_date) and (end_date is None or date <= end_date)

def stratified_sample_by_time(data, time_column, freq, sample_ratio, datetime_format="%Y-%m-%d %H:%M:%S"):
    data['temp_time_column'] = pd.to_datetime(data[time_column], format=datetime_format, errors='coerce')
    sampled_data = data.groupby(pd.Grouper(key='temp_time_column', freq=freq)).apply(lambda x: x.sample(frac=sample_ratio) if len(x) > 0 else x)
    sampled_data.reset_index(drop=True, inplace=True)
    del sampled_data['temp_time_column']
    return sampled_data

def sample_by_datetime(directory, time_column, freq, sample_ratio, datetime_format="%Y-%m-%d %H:%M:%S", start_date=None, end_date=None):
    all_sampled_data = pd.DataFrame() 
    start_date = safe_str_to_date(start_date, datetime_format) if start_date else None
    end_date = safe_str_to_date(end_date, datetime_format) if end_date else None
    for file_name in os.listdir(directory):
        if not file_name.endswith('.parquet'):
            continue
        file_path = os.path.join(directory, file_name)
        data = pd.read_parquet(file_path) 
        data['temp_time_column'] = pd.to_datetime(data[time_column], format=datetime_format, errors='coerce')
        if start_date or end_date:
            data = data[(data['temp_time_column'].notnull()) & (data['temp_time_column'].between(start_date, end_date))]
        stratified_data = stratified_sample_by_time(data, time_column, freq, sample_ratio, datetime_format)        
        all_sampled_data = pd.concat([all_sampled_data, stratified_data], ignore_index=True)
    return all_sampled_data

def count_total_rows(directory):
    total_rows = 0
    files_count = 0
    for file_name in [f for f in os.listdir(directory) if f.endswith('.parquet')]:
        total_rows += len(pd.read_parquet(os.path.join(directory, file_name)))
        files_count += 1
    print(f'Total number of files processed {files_count} containing: {total_rows} rows')
    return total_rows