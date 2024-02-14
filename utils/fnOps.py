from datetime import datetime
import os
import pandas as pd

from clearml import Task, Dataset as ClearMLDataset, Model as ClearMLModel, Logger, OutputModel, InputModel

from utils.fnProcessing import read_config
from generative_text.general_tnn.train import train_model, CustomSchedule, TrainTextGenerator
from generative_text.general_tnn.process import main
from generative_text.general_tnn.tnn import TransformerBlock, TokenAndPositionEmbedding
from generative_text.general_tnn.evaluate import TextGenerator

from utils.fnCloud.S3Handler import S3Handler
from generative_text.paired_tnn.process import process_paired_data
from generative_text.paired_tnn.train import prepare_model_training, PairedTNNTextGenerator

from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.framework import ops
import tensorflow_io

config_path='./utils/fnCloud/config-cloud.ini'
config_params = read_config(section='aws_credentials',config_path=config_path)
config_s3_info = read_config(section='s3_information',config_path=config_path)

class ClearMLOps:
    def __init__(self, config_path='./generative_text/config.ini'):
        # Swap between params-general or params-paired
        self.config_params = read_config(section='params-general', config_path=config_path)
        self.clearml_params = read_config(section='clearml', config_path=config_path)
        self.set_creds_connect()

    def set_creds_connect(self):
        api_host = os.getenv('CLEARML_API_HOST', self.clearml_params.get('api_host'))
        web_host = os.getenv('CLEARML_WEB_HOST', self.clearml_params.get('web_host'))
        files_host = os.getenv('CLEARML_FILES_HOST', self.clearml_params.get('files_host'))
        key = os.getenv('CLEARML_KEY', self.clearml_params.get('key'))
        secret = os.getenv('CLEARML_SECRET', self.clearml_params.get('secret'))
        os.environ['AWS_ACCESS_KEY_ID'] = config_params['aws_access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = config_params['aws_secret_access_key']
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        if not all([api_host, web_host, files_host, key, secret]):
            raise ValueError("Missing necessary ClearML credentials. Please set them in environment variables or config file.")
        Task.set_credentials(api_host=api_host, web_host=web_host, files_host=files_host, key=key, secret=secret)

    def list_datasets(self, project_name=None):
        datasets = ClearMLDataset.list_datasets(dataset_project=project_name)
        for dataset in datasets:
            print(f"ID: {dataset['id']}, Name: {dataset['name']}, Project: {dataset['project']}, Tags: {dataset.get('tags', [])}")

    def initialize_clearml(self, task_name, task_type):
        try:
            task = Task.init(project_name=self.clearml_params['clearml_project_name'],
                            task_name=task_name,
                            output_uri=self.clearml_params['clearml_output_uri'],
                            task_type=task_type)
            task.connect(self.config_params)
            return task
        except Exception as e:
            print(f"Failed to initialize ClearML task: {e}")
            return None

    def save_dataset_to_local(self, df, dir_path, filename):
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        try:
            df.to_parquet(os.path.join(dir_path, filename))
            print(f"Saved DataFrame to {filename}")
        except Exception as e:
            print(f"Failed to save DataFrame to {filename}: {e}")
            
    def upload_dataset_to_clearml(self, dataset_uri, dataset_name, task_name):
        try:
            clearml_dataset = ClearMLDataset.create(dataset_project=self.clearml_params['clearml_project_name'], dataset_name=dataset_name, output_uri=self.clearml_params['clearml_output_uri'], description=f"{task_name}")
            clearml_dataset.add_files(dataset_uri)
            clearml_dataset.finalize(auto_upload=True)
            print(f"Uploaded {dataset_name} to ClearML.")
        except Exception as e:
            print(f"Failed to upload dataset to ClearML: {e}")

    def load_dataset(self, dataset_id):
        dataset = ClearMLDataset.get(dataset_id=dataset_id, alias=f'{dataset_id}')
        local_copy_path = dataset.get_local_copy()
        try:
            df = pd.read_parquet(local_copy_path)
            return df
        except Exception as e:
            print(f"Failed to load dataset from {local_copy_path}: {e}")
            return None

    def get_latest_datasets(self, project_name, base_name='context'):
        datasets = self.list_datasets(self.clearml_params['clearml_project_name'])
        filtered_datasets = [d for d in datasets if d['Name'].startswith(base_name)]
        sorted_datasets = sorted(filtered_datasets, key=lambda x: x['created'], reverse=True)
        latest_test = sorted_datasets[0] if sorted_datasets else None
        return latest_test

    def generate_file_name(self, ds_name, task_name, unique_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{task_name}_{ds_name}_{unique_id}_{timestamp}.parquet"
        return file_name

    def process_data(self, training_data, dataset_name, task_name):
        task = self.initialize_clearml(task_name, Task.TaskTypes.data_processing)
        if task:
            os.makedirs(self.process_config['dir_loc'], exist_ok=True)
            filename = self.generate_file_name(dataset_name, task.name, task.id)
            self.save_dataset_to_local(training_data, self.process_config['dir_loc'], filename)
            self.upload_dataset_to_clearml(os.path.join(self.process_config['dir_loc'], filename), f'{dataset_name}_{task.id}', task_name)
            task.close()

    def get_training_set(self, task_name):
        dataset_info = self.get_latest_datasets(project_name= self.clearml_params['clearml_project_name'])
        combined_dataset = self.load_dataset(dataset_info['id'])
        train_ds, val_ds, test_ds, combined_vocab, tokenizer = main(combined_dataset, input_col='text', clean_col='text')
        return train_ds, val_ds, test_ds, combined_vocab

    def list_models(self, project_name):
        project_name = project_name 
        models = ClearMLModel.query_models(project_name=project_name)
        print(f"Models in project {project_name}:")
        for model in models:
            print(f"Model ID: {model.id}, Name: {model.name}")
            
    def load_model(self, model_id): 
        model = ClearMLModel(model_id=model_id)
        print(f"Model ID: {model.id}")
        print(f"Name: {model.name}")
        print(f"URL: {model.url}")
        local_weights_path = model.get_weights()
        print(f"Weights downloaded to: {local_weights_path}")

    def remove_clearml_datasets(self, dataset_ids=None):
        if dataset_ids is None:
            all_datasets = Dataset.list_datasets(dataset_project=self.clearml_params['clearml_project_name'])
            for dataset in all_datasets:
                try:
                    ds = Dataset.get(dataset_id=dataset['id'])
                    ds.delete(dataset_id = ds.id)
                except Exception as e:
                    print(f"Failed to delete dataset: {dataset['name']} (ID: {dataset['id']}). Reason: {e}")
        else:
            for dataset_id in dataset_ids:
                try:
                    ds = Dataset.get(dataset_id=dataset_id)
                    ds.delete(dataset_id = ds.id)
                except Exception as e:
                    print(f"Failed to delete dataset with ID: {dataset_id}. Reason: {e}")

class ClearMLOpsTraining(ClearMLOps):
    def __init__(self, clearml_params, config_params, config_aws, config_path='./generative_text/config.ini'):
        super().__init__(config_path)
        self.clearml_params = clearml_params
        self.config_params = config_params
        self.config_aws = config_aws
        self.combined_params = {**self.config_params, **self.clearml_params}
        self.configure_aws_credentials()
        self.s3_handler = S3Handler(self.config_aws, config_s3_info)

    def configure_aws_credentials(self):
        os.environ['AWS_ACCESS_KEY_ID'] = self.config_aws['aws_access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.config_aws['aws_secret_access_key']
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        
    def get_callbacks_general(self, task_output_uri, combined_vocab, tokenizer):
        best_model_path = os.path.join(task_output_uri, 'best_model.keras')
        return [
            ModelCheckpoint(filepath=best_model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
            TrainTextGenerator(index_to_word=combined_vocab, tokenizer=tokenizer),
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=False)]

    def get_callbacks_paired(self, task_output_uri, tokenizer):
        best_model_path = os.path.join(task_output_uri, 'best_model.keras')
        return [
            ModelCheckpoint(filepath=best_model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
            PairedTNNTextGenerator(tokenizer=tokenizer),
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=False)]

    def train_and_evaluate(self, model, train_ds, val_ds, test_ds, callbacks, task):
        logger = task.get_logger()
        task_output_uri = self.clearml_params['clearml_output_uri']
        best_model_path = os.path.join(task_output_uri, 'best_model.keras')
        history = model.fit(train_ds, epochs=int(self.combined_params['epochs']), validation_data=val_ds, callbacks=callbacks)
        
        # Log training history
        for epoch in range(len(history.epoch)):
            for metric, values in history.history.items():
                value = values[epoch]
                stage = 'train_loss' if 'val' not in metric else 'val_loss'
                metric_name = metric.replace('val_', '')
                logger.report_scalar(title=metric_name, series=stage, value=value, iteration=epoch)

        # Evaluate on test set
        test_metrics = model.evaluate(test_ds)
        if isinstance(test_metrics, float):
            test_metrics = [test_metrics]
        for metric_name, value in zip(model.metrics_names, test_metrics):
            print(f"{metric_name}: {value}")
        for epoch in range(len(history.epoch)):
            for metric_name, value in zip(model.metrics_names, test_metrics):
                logger.report_scalar(title=metric_name, series="test", value=value, iteration=epoch)
        print(f"Best model saved at {best_model_path}")
        # Upload the best model to S3
        s3_model_key = f"models/{task.id}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_best_model.keras"
        self.s3_handler._upload_file(best_model_path, s3_model_key)
        print(f"Best model uploaded to S3 at {s3_model_key}")
        logger.flush()

    def run_clearml_training_task(self, task_name, dataset_type='general', dataset_id=None, load_model='new', model_id=None):
        output_uri = self.clearml_params['clearml_output_uri']
        task = Task.init(project_name=self.combined_params['clearml_project_name'], task_name=task_name, auto_connect_frameworks={'tensorboard': True}, output_uri=output_uri, task_type=Task.TaskTypes.training)
        task.connect(self.combined_params)
        if task:
            combined_dataset = self.load_dataset(dataset_id)
            
            ## General data
            if dataset_type == 'general':
                if dataset_id:
                    train_ds, val_ds, test_ds, combined_vocab, tokenizer = main(combined_dataset, input_col='text', clean_col='text')
                else:
                    train_ds, val_ds, test_ds = self.get_latest_datasets(self.combined_params['clearml_project_name'])['id']
                custom_objects = {
                    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                    'TransformerBlock': TransformerBlock,
                    'CustomSchedule': CustomSchedule}
                output_model = OutputModel(task=task, framework='Keras')
                if load_model == 'train':
                    input_model = InputModel(model_id=model_id)
                    local_model_path = input_model.get_local_copy(force_download=False)
                    model = train_model(preload_model=True, model_path=local_model_path)
                elif load_model == 'new':
                    model = train_model(preload_model=False)
                callbacks = self.get_callbacks_general(output_uri, combined_vocab, tokenizer)
                model_filename = os.path.join(output_uri, f"{self.combined_params['model_name_general']}_{task.id}.keras")
                
            ## Paired data             
            if dataset_type == 'paired':
                custom_objects = {
                    'TransformerBlock': TransformerBlock,
                    'CustomSchedule': CustomSchedule}
                output_model = OutputModel(task=task, framework='Keras')
                if load_model == 'train':
                    input_model = InputModel(model_id=model_id)
                    local_model_path = input_model.get_local_copy(force_download=False)
                    model,train_ds, val_ds, test_ds, vocab, tokenizer = prepare_model_training(combined_dataset, preload_model=True, model_path=local_model_path)
                elif load_model == 'new':
                    model,train_ds, val_ds, test_ds, vocab, tokenizer = prepare_model_training(combined_dataset, preload_model=False)
                callbacks = self.get_callbacks_paired(output_uri, tokenizer)
                model_filename = os.path.join(output_uri, f"{self.combined_params['model_name_paired']}_{task.id}.keras")
            self.train_and_evaluate(model, train_ds, val_ds, test_ds, callbacks, task)
            output_model.update_weights(weights_filename=model_filename)
            task.close()