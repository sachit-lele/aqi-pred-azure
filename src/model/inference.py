import os
import mlflow
import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load the ML model from the specified path.
    """
    logger.info(f"Loading model from: {model_path}")
    try:
        model = mlflow.pyfunc.load_model(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model. Error: {e}")
        raise

def make_predictions(model, input_data):
    """
    Make predictions using the loaded model and input data.
    """
    logger.info("Making predictions...")
    try:
        predictions = model.predict(input_data)
        logger.info("Predictions completed successfully.")
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed. Error: {e}")
        raise

def main(model_path, input_file, output_file):
    """
    Main function to handle the inference process.
    """
    # Load the model
    model = load_model(model_path)

    # Load input data
    logger.info(f"Loading input data from: {input_file}")
    try:
        data = pd.read_csv(input_file)
        logger.info(f"Input data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to read input file. Error: {e}")
        raise

    # Make predictions
    predictions = make_predictions(model, data)

    # Prepare output DataFrame
    output_df = pd.DataFrame(predictions, columns=['Predictions'])
    output_df['Source_File'] = os.path.basename(input_file)

    # Save predictions to CSV
    logger.info(f"Saving predictions to: {output_file}")
    try:
        output_df.to_csv(output_file, index=False)
        logger.info("Predictions saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save predictions. Error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Inference Script for ML Model")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model directory')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input CSV file with test data')
    parser.add_argument('--output-file', type=str, default='predictions.csv', help='Path to save the output predictions')

    args = parser.parse_args()

    main(args.model_path, args.input_file, args.output_file)