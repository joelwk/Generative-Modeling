# import necessary modules
'''
https://github.com/aws/amazon-sagemaker-examples/blob/main/ingest_data/ingest-data-types/ingest_tabular_data.ipynb
'''
import os
import boto3
import logging
import configparser
from botocore.exceptions import BotoCoreError, ClientError

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Read configuration
from generative_text.general_tnn.utils.fnProcessing import read_config

config_path='./generative_text/general_tnn/utils/fnAWS/config-AWS.ini'
config_params = read_config(section='aws_credentials',config_path=config_path)
config_s3_info = read_config(section='s3_information',config_path=config_path)

logger = logging.getLogger(__name__)

bucket_name = config_s3_info['s3_bucket'], 
s3_bucket_data = config_s3_info['s3_bucket_data']
aws_access_key_id = config_params['aws_access_key_id']
aws_secret_access_key = config_params['aws_secret_access_key']

class S3Handler:
    def __init__(self, aws_credentials, s3_info, local_data_path='../data-s3', region_name='us-east-1'):
        # Correctly assign bucket_name from s3_info dictionary
        self.bucket_name = s3_info['s3_bucket']
        self.local_path = os.path.normpath(local_data_path) + os.sep
        self.region_name = region_name
        
        # Setup session and s3 client
        if aws_credentials.get('profile_name'):
            session = boto3.Session(profile_name=aws_credentials['profile_name'])
        else:
            session = boto3.Session(aws_access_key_id=aws_credentials['aws_access_key_id'],
                                    aws_secret_access_key=aws_credentials['aws_secret_access_key'],
                                    region_name=self.region_name)
        self.s3 = session.client('s3')
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def get_s3_client(self):
        return self.s3

    def file_exists_in_s3(self, s3_key):
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3.exceptions.ClientError:
            return False
            
    def upload_dir(self, dir_key):
        for root, _, files in os.walk(self.local_path):
            print(root)
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, self.local_path)
                s3_key = os.path.join(dir_key, relative_path).replace(os.sep, '/')
                print(f"Bucket name: {self.bucket_name}, Type: {type(self.bucket_name)}")

                print(s3_key)
                if not local_file.endswith('/') and not local_file.endswith('\\'):
                    if not self.file_exists_in_s3(s3_key):
                        self._upload_file(local_file, s3_key)
                    else:
                        print(f"Skipping {s3_key}, already exists in S3.")

    def download_dir(self, dir_key):
        self.prefix = dir_key  # define the prefix here
        paginator = self.s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=dir_key):
            for file in result.get('Contents', []):
                self._download_file(file['Key'])

    def _download_file(self, key):
        if key.endswith('/'):  # If key is a directory, skip
            return
        relative_file_path = key.replace(self.prefix, '', 1).lstrip('/')
        local_file = os.path.join(self.local_path, relative_file_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        self.s3.download_file(self.bucket_name, key, local_file)
        logger.info(f"Downloaded file {key} to {local_file}")

    def _upload_file(self, local_file, key):
        self.s3.upload_file(local_file, self.bucket_name, key)
        logger.info(f"Uploaded file {local_file} to {self.bucket_name}/{key}")

    def cleanup_s3_file(self, key):
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted file {key} from bucket {self.bucket_name}")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error deleting file {key} from bucket {self.bucket_name}. Error: {e}")
            raise

    def update_file(self, df, local_path):
        try:
            df.to_csv(local_path, mode='w', header=False, index=False)
            logger.info(f"Updated file {local_path}")
        except IOError as e:
            logger.error(f"Failed to update file {local_path}. Error: {e}")
            raise

    def create_dir(self, dir_key):
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=(dir_key + '/'))
            logger.info(f"Created directory {dir_key} in bucket {self.bucket_name}")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error creating directory {dir_key} in bucket {self.bucket_name}. Error: {e}")
            raise

    def delete_dir(self, dir_key):
        try:
            objects = self.s3.list_objects(Bucket=self.bucket_name, Prefix=dir_key)
            for obj in objects.get('Contents', []):
                self.s3.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
            logger.info(f"Deleted directory {dir_key} from bucket {self.bucket_name}")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error deleting directory {dir_key} from bucket {self.bucket_name}. Error: {e}")
            raise

    def move_file(self, old_key, new_key):
        try:
            self.s3.copy_object(Bucket=self.bucket_name, CopySource={'Bucket': self.bucket_name, 'Key': old_key}, Key=new_key)
            self.s3.delete_object(Bucket=self.bucket_name, Key=old_key)
            logger.info(f"Moved file from {old_key} to {new_key} in bucket {self.bucket_name}")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error moving file from {old_key} to {new_key} in bucket {self.bucket_name}. Error: {e}")
            raise

    def list_all_files(self, prefix=''):
        paginator = self.s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in result.get('Contents', []):
                file_key = obj['Key']
                file_size = obj['Size']
                last_modified = obj['LastModified']
                etag = obj['ETag']
                print(f"File: {file_key} Size: {file_size} Last Modified: {last_modified} ETag: {etag}\n")