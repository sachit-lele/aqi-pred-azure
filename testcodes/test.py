import pandas as pd
import argparse
import pickle
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logging.info(f"Model loaded from {model_path}")
    return model

def load_data(data_path):
    data = pd.read_csv(data_path)
    logging.info(f"Data loaded from {data_path} with shape {data.shape}")
    return data

def predict(model, data):
    try:
        predictions = model.predict(data)
        logging.info("Predictions completed.")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise
    return predictions

def save_predictions(predictions, output_path):
    predictions_df = pd.DataFrame(predictions, columns=["Prediction"])
    predictions_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Local Inference Script")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the local model (.pkl file)')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the local dataset (.csv file)')
    parser.add_argument('--output-path', type=str, default='predictions.csv', help='Path to save predictions')
    args = parser.parse_args()

    # Load model and data
    model = load_model(args.model_path)
    data = load_data(args.data_path)

    # Run predictions
    predictions = predict(model, data)

    # Save predictions
    save_predictions(predictions, args.output_path)

if __name__ == '__main__':
    main()
