import html
import random
import gzip
import string
import re
import pickle
import io
import html
import configparser
import boto3
import pandas as pd
import numpy as np
from unicodedata import normalize
import os
import random
import glob

import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read('./generative_text/general_tnn/utils/fnAWS/config-AWS.ini')
s3_event = config["s3_event"]
s3_event_processing = config["s3_event_processing"]
s3_bucket_data = s3_event['s3_bucket_data']
s3_bucket_processed = s3_event['s3_bucket_processed']
s3_file_ext = config["s3_file_ext"]
processed_data_gz = s3_file_ext['processed_data_gz']
s3_bucket_batchprocessed = s3_event['s3_bucket_batchprocessed']
s3_c = boto3.client('s3')

def get_random_batch_data_from_s3(bucket, s3_prefix, sample_ratio=None, file_ratio=None):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    batch_objects = [item['Key'] for item in response['Contents'] if item['Key'].startswith(s3_prefix + 'batch_')]
    if not batch_objects:
        print("No 'batch_' files found.")
        return None
    if file_ratio is not None:
        batch_objects = random.sample(batch_objects, int(len(batch_objects) * file_ratio))
    all_data = []
    # Go through all batch_objects
    for key in batch_objects:
        # Get the S3 object
        obj = s3.get_object(Bucket=bucket, Key=key)
        # Get content of the object
        gz_data = obj['Body'].read()
        # Decompress the gzip stream and load the pickled DataFrame
        with gzip.GzipFile(fileobj=io.BytesIO(gz_data)) as gzipfile:
            data = pickle.load(gzipfile)
        # Sample the data if sample_ratio is provided
        if sample_ratio is not None:
            data = data.sample(frac=sample_ratio)
        all_data.append(data)
    # Concatenate all sampled dataframes
    sampled_df = pd.concat(all_data, ignore_index=True)
    return sampled_df

def load_processed_data(file_path):
    buffer = io.BytesIO()
    s3_c.download_fileobj(s3_bucket_name, file_path, buffer)
    buffer.seek(0)
    with gzip.GzipFile(fileobj=buffer, mode='r') as f:
        return pickle.load(f)

def get_random_sample_csv_from_s3(bucket, s3_prefix, sample_ratio=None):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    csv_objects = [item['Key'] for item in response['Contents'] if item['Key'].endswith('.csv')]
    if not csv_objects:
        print("No CSV files found.")
        return None
        
    # Select a random file
    random_file_key = random.choice(csv_objects)
    temp_file = "temp.csv"
    s3.download_file(bucket, random_file_key, temp_file)
    data = pd.read_csv(temp_file)
    if sample_ratio is not None:
        data = data.sample(frac=sample_ratio)
    os.remove(temp_file)
    return data

def get_random_sample_parquet_from_local(dir_pattern, sample_ratio=None):
    parquet_files = glob.glob(dir_pattern)
    if not parquet_files:
        print("No parquet files found.")
        return None
    
    # Select a random file
    random_file_path = random.choice(parquet_files)
    data = pd.read_parquet(random_file_path)
    
    if sample_ratio is not None:
        data = data.sample(frac=sample_ratio)
    
    return data

def load_processed_sample_s3(bucket, s3_prefix, file_ratio=None, sample_ratio=None):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    parquet_objects = [item['Key'] for item in response['Contents'] if item['Key'].endswith('.parquet')]
    if not parquet_objects:
        print("No parquet files found.")
        return None
    # Sample a fraction of the files
    if file_ratio is not None:
        parquet_objects = random.sample(parquet_objects, int(len(parquet_objects) * file_ratio))
    dfs = []
    for file_key in parquet_objects:
        # Use a temporary file to download and load each parquet file
        temp_file = "temp.parquet"
        s3.download_file(bucket, file_key, temp_file)
        df = pd.read_parquet(temp_file)
        # Sample a fraction of the rows from each file
        if sample_ratio is not None:
            df = df.sample(frac=sample_ratio)
        dfs.append(df)
        # Remove the temporary file
        os.remove(temp_file)
    # Concatenate all dataframes
    data = pd.concat(dfs)
    return data