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
import logging
import pandas as pd
import configparser
from utils.fnCloud.S3Handler import S3Handler
from utils.fnProcessing import read_config
from utils.fnSampling import stratified_sample_by_time

config_path='./utils/fnCloud/config-cloud.ini'
config_params = read_config(section='aws_credentials',config_path=config_path)
config_s3_info = read_config(section='s3_information',config_path=config_path)

logger = logging.getLogger(__name__)

def get_random_batch_data_from_s3(bucket, s3_prefix, sample_ratio=None, file_ratio=None, stratify=True):
    connector = S3Handler(config_params, config_s3_info)
    s3_client = connector.get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    batch_objects = [item['Key'] for item in response['Contents'] if item['Key'].startswith(s3_prefix + 'batch_')]
    if not batch_objects:
        print("No 'batch_' files found.")
        return None
    if file_ratio is not None:
        batch_objects = random.sample(batch_objects, int(len(batch_objects) * file_ratio))
    all_data = []
    for key in batch_objects:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        gz_data = obj['Body'].read()
        with gzip.GzipFile(fileobj=io.BytesIO(gz_data)) as gzipfile:
            data = pickle.load(gzipfile)
        if sample_ratio is not None:
            if stratify:
                data = stratified_sample_by_time(data, 'posted_date_time', 'H', sample_ratio)
            else:
                data = data.sample(frac=sample_ratio)
        all_data.append(data)
    sampled_df = pd.concat(all_data, ignore_index=True)
    return sampled_df

def load_processed_data(data,s3_prefix):
    connector = S3Handler(config_params, config_s3_info)
    s3_client = connector.get_s3_client()
    buffer = io.BytesIO()
    s3_client.download_fileobj(config_s3_info['s3_bucket'], s3_prefix, buffer)
    buffer.seek(0)
    with gzip.GzipFile(fileobj=buffer, mode='r') as f:
        return pickle.load(f)

def get_random_sample_csv_from_s3(bucket, s3_prefix, sample_ratio=None, stratify=True):
    connector = S3Handler(config_params, config_s3_info)
    s3_client = connector.get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    csv_objects = [item['Key'] for item in response['Contents'] if item['Key'].endswith('.csv')]
    if not csv_objects:
        print("No CSV files found.")
        return None
    random_file_key = random.choice(csv_objects)
    temp_file = "temp.csv"
    s3_client.download_file(bucket, random_file_key, temp_file)
    data = pd.read_csv(temp_file)
    if sample_ratio is not None:
        stratify_column = None
        if 'posted_date_time' in data.columns:
            stratify_column = 'posted_date_time'
        elif 'comment_posted_date_time' in data.columns:
            stratify_column = 'comment_posted_date_time'
        if stratify and stratify_column:
            data = stratified_sample_by_time(data, stratify_column, 'H', sample_ratio)
        else:
            data = data.sample(frac=sample_ratio)
    os.remove(temp_file)
    return data

def get_random_sample_parquet_local(dir_pattern, sample_ratio=None, stratify=False):
    parquet_files = glob.glob(dir_pattern)
    if not parquet_files:
        print("No parquet files found.")
        return None
    random_file_path = random.choice(parquet_files)
    data = pd.read_parquet(random_file_path)
    if sample_ratio is not None:
        data = data.sample(frac=sample_ratio)
    return data

def load_processed_parquet_s3(bucket, s3_prefix, file_ratio=None, sample_ratio=None, stratify=False):
    connector = S3Handler(config_params, config_s3_info)
    s3_client = connector.get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    parquet_objects = [key for key in response if key.endswith('.parquet')]
    if not parquet_objects:
        print("No parquet files found.")
        return None
    if file_ratio is not None:
        parquet_objects = random.sample(parquet_objects, int(len(parquet_objects) * file_ratio))
    dfs = []
    for file_key in parquet_objects:
        temp_file = "temp.parquet"
        connector._download_file(bucket, file_key, temp_file)
        df = pd.read_parquet(temp_file)
        if sample_ratio is not None:
            if stratify and 'posted_date_time' in df.columns:
                df = stratified_sample_by_time(df, 'posted_date_time', 'H', sample_ratio)
            else:
                df = df.sample(frac=sample_ratio)
        dfs.append(df)
        os.remove(temp_file)
    data = pd.concat(dfs)
    return data