
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Initialize the model variable
model = None

def init():
    global model
    # Load the model from the specified path
    model_path = 'path/to/your/saved/model/file.joblib'  # Update this to your actual model path
    model = joblib.load(model_path)
    print("Model loaded successfully.")

def preprocess_input(data):
    # Preprocess the input data in a similar way as done during training
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date').drop(["AQI_Bucket", 'Xylene', 'Benzene', 'O3'], axis=1)

    # Handle outliers and missing values, if needed
    for column in data.select_dtypes(include='number').columns:
        Q1, Q3 = data[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        data[column] = np.clip(data[column], lower_bound, upper_bound)
    
    # Other transformations or feature scaling can be added here
    return data

def run(raw_data):
    # Parse the input data
    try:
        data = pd.DataFrame(json.loads(raw_data)['data'])
    except Exception as e:
        return json.dumps({"error": f"Failed to parse input data: {str(e)}"})

    # Preprocess the input data
    processed_data = preprocess_input(data)

    # Make predictions using the loaded model
    try:
        predictions = model.predict(processed_data)
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": f"Failed to make predictions: {str(e)}"})
