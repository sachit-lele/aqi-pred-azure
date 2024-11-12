# src/models/inference.py

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

def save_predictions(predictions, output_path):
    predictions_df = pd.DataFrame(predictions, columns=["Prediction"])
    predictions_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

def main(model_path, data_path, output_path="predictions.csv"):
    # Load model and data
    model = load_model(model_path)
    data = load_data(data_path)

    # Instead of running predictions, append the number 103 to the output
    predictions = append_number(data, number=103)

    # Save predictions
    save_predictions(predictions, output_path)

def load_model(model_path):
    """
    Dummy load_model function to keep the model code intact.
    """
    try:
        # Normally, you would load your model here
        # model = joblib.load(model_path)
        # logging.info(f"Model loaded from {model_path}")
        # return model
        
        # Since we're not using the model, return None or a dummy object
        logging.info(f"Model loading skipped. Appending instead.")
        return None
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

def load_data(data_path):
    """
    Loads data from a CSV file.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {data_path}: {e}")
        raise

def append_number(data, number=103):
    """
    Appends the specified number to the predictions.
    """
    try:
        # Create a list with the number for each row in the data
        predictions = [number] * len(data)
        logging.info(f"Appended number {number} to predictions.")
        return predictions
    except Exception as e:
        logging.error(f"Failed to append number {number} to predictions: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the input data CSV file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions")
    args = parser.parse_args()

    main(args.model_path, args.data_path, args.output)