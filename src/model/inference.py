# batch_inference.py

import os
import joblib
import pandas as pd
import argparse
import logging

from azureml.core import Workspace, Dataset, Model, Experiment, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch Inference Pipeline Script")
    parser.add_argument('--workspace-config', type=str, default='config.json', help='Path to Azure ML workspace config file')
    args = parser.parse_args()

    # Connect to Azure ML Workspace
    ws = Workspace.get(
    name='myWorkSpace',
    subscription_id='946966e0-6b02-4b92-89d2-c2e3bb3604c7',
    resource_group='myResource',
    )

    # Define Experiment
    experiment_name = 'batch_inference_experiment'
    experiment = Experiment(workspace=ws, name=experiment_name)
    logging.info(f"Using Experiment: {experiment_name}")

    # Define or get Compute Target
    compute_name = 'lelesachitcluster'
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        logging.info(f"Found existing compute target: {compute_name}")
    except Exception as e:
        logging.error(f"Compute target '{compute_name}' not found: {e}")
        raise

    # Get Datastore
    datastore_name = 'aqi_pred_datastore'
    try:
        datastore = Datastore.get(ws, datastore_name)
        logging.info(f"Using Datastore: {datastore_name}")
    except Exception as e:
        logging.error(f"Datastore '{datastore_name}' not found: {e}")
        raise

    # Get Dataset
    dataset_name = 'test_dataset'
    try:
        dataset = Dataset.File.from_files(path=(datastore, 'test.csv'))
        logging.info(f"Using Dataset: {dataset_name}")
    except Exception as e:
        logging.error(f"Dataset '{dataset_name}' creation failed: {e}")
        raise

    # Get Registered Model
    model_name = 'dectree_aqi_model'
    try:
        model = Model(ws, name=model_name)
        logging.info(f"Retrieved Model: {model_name}, version: {model.version}")
    except Exception as e:
        logging.error(f"Model '{model_name}' not found: {e}")
        raise

    # Define Environment
    env_name = 'CustomEnv01'
    try:
        env = Environment.get(workspace=ws, name=env_name)
        logging.info(f"Using Environment: {env_name}")
    except Exception as e:
        logging.error(f"Environment '{env_name}' not found: {e}")
        raise

    # Define Inference Configuration
    inference_config = InferenceConfig(entry_script="src/model/inference.py",
                                       environment=env)
    logging.info("InferenceConfig created.")

    # Define Pipeline Output
    output_datastore_name = 'aqi_pred_datastore'
    try:
        output_datastore = Datastore.get(ws, output_datastore_name)
        logging.info(f"Using Output Datastore: {output_datastore_name}")
    except Exception as e:
        logging.error(f"Output Datastore '{output_datastore_name}' not found: {e}")
        raise

    predictions_output = PipelineData("predictions", datastore=output_datastore)
    logging.info("PipelineData for predictions defined.")

    # Define Pipeline Step
    inference_step = PythonScriptStep(
        name="Batch Inference",
        script_name="inference.py",
        arguments=[
            "--input-data", dataset.as_named_input('input_dataset').as_mount(),
            "--model-name", model.name,
            "--output", predictions_output
        ],
        outputs=[predictions_output],
        compute_target=compute_target,
        source_directory='src/model',  # Directory containing inference.py
        runconfig=inference_config,
        allow_reuse=False
    )
    logging.info("PythonScriptStep for Batch Inference defined.")

    # Create Pipeline
    pipeline = Pipeline(workspace=ws, steps=[inference_step])
    logging.info("Pipeline created.")

    # Submit Pipeline Run
    logging.info("Submitting pipeline run...")
    pipeline_run = experiment.submit(pipeline)
    pipeline_run.wait_for_completion(show_output=True)
    logging.info("Pipeline run completed.")

    # Download Predictions
    try:
        predictions_output.download(target_path='predictions_output', overwrite=True)
        logging.info("Predictions downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download predictions: {e}")
        raise

if __name__ == '__main__':
    main()