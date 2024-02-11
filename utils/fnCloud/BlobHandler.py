from azure.storage.blob import BlobServiceClient, BlobClient
import os
import pandas as pd
import logging
import io
import gzip
import pickle
import glob
import random

from utils.fnProcessing import read_config

config_path='./utils/fnCloud/config-cloud.ini'
config_params = read_config(section='azure_credentials',config_path=config_path)
config_s3_info = read_config(section='s3_information',config_path=config_path)

logger = logging.getLogger(__name__)

bucket_name = config_s3_info['s3_bucket'], 
s3_bucket_data = config_s3_info['s3_bucket_data']
aws_access_key_id = config_params['aws_access_key_id']
aws_secret_access_key = config_params['aws_secret_access_key']

class BlobHandler:
    def __init__(self, connection_string, container_name, local_data_path='../data-azure',):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.local_path = os.path.normpath(local_data_path) + os.sep
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def get_blob_client(self):
        return self.blob_service_client

    def file_exists_in_blob(self, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)
        try:
            blob_client.get_blob_properties()
            return True
        except Exception as e:
            return False
            
    def upload_dir(self, azure_dir_key):
        for root, _, files in os.walk(self.local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, self.local_path)
                blob_name = os.path.join(azure_dir_key, relative_path).replace(os.sep, '/')
                if not self.file_exists_in_blob(blob_name):
                    self._upload_file(local_file, blob_name)
                else:
                    print(f"Skipping {blob_name}, already exists in Blob Storage.")

    def _upload_file(self, local_file, blob_name):
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        self.logger.info(f"Uploaded file {local_file} to Blob Storage as {blob_name}")

    def download_blob(self, blob_name, local_file):
        blob_client = self.container_client.get_blob_client(blob_name)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        with open(local_file, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        self.logger.info(f"Downloaded blob {blob_name} to {local_file}")
        
        ###TODO: Complete the class and test