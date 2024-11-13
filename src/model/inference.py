# batch_inference.py

import os
import pandas as pd
import argparse
import logging


from azureml.core import Workspace, Dataset, Experiment, Datastore
from azureml.core.compute import ComputeTarget
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
    def generate_prediction():
        df = pd.DataFrame({"prediction": [220]})
        output_path = os.path.join("outputs", "predictions.csv")
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    # Run the dummy prediction function instead of model inference
    generate_prediction()

    # Create and Submit Pipeline
    pipeline = Pipeline(workspace=ws, steps=[])
    logging.info("Pipeline created with prediction.")

    pipeline_run = experiment.submit(pipeline)
    logging.info("Pipeline run completed with prediction.")

if __name__ == '__main__':
    main()
