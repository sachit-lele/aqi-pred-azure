import numpy as np
import pandas as pd
import logging
from warnings import filterwarnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import argparse
from sklearn.metrics import r2_score
from azureml.core import Dataset, Run, Workspace, Experiment, Environment, Datastore
from azureml.core.model import Model
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.script_run_config import ScriptRunConfig
import os


logging.basicConfig(level=logging.INFO)
filterwarnings('ignore')

ws = Workspace.get(
    name='myWorkSpace',
    subscription_id='946966e0-6b02-4b92-89d2-c2e3bb3604c7',
    resource_group='myResource'
)

experiment = Experiment(workspace=ws, name='air-quality-index-experiment01')

clustername = 'lelesachitcluster'
compute_target = ComputeTarget(workspace=ws, name=clustername)

env = Environment.get(workspace=ws, name='CustomEnv01')

runconfig = RunConfiguration()
runconfig.environment = env
runconfig.target = compute_target

def main(data_path):
    # Get the experiment run context
    run = Run.get_context()

    # Load dataset using Azure ML Dataset API
    dataset = Dataset.File.from_files(data_path)
    df_city_day = dataset.to_pandas_dataframe()

    # Convert Date column to datetime and sort by Date
    df_city_day['Date'] = pd.to_datetime(df_city_day['Date'], format='%Y-%m-%d')
    df_city_day = df_city_day.sort_values(by='Date')

    # Drop AQI_Bucket column
    df_city_day = df_city_day.drop(["AQI_Bucket"], axis=1)

    # Replace outliers with quartile values
    def replace_outliers_with_quartiles(df):
        for column in df.select_dtypes(include=['number']).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].apply(lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x))
        return df

    # Replace outliers
    df_city_day = replace_outliers_with_quartiles(df_city_day)

    # Drop unwanted columns
    df_city_day = df_city_day.drop(columns=['Xylene', 'Benzene', 'O3'])

    # Filter rows with non-null AQI values
    df_full = df_city_day[df_city_day['AQI'].notna()]

    # Extract Year from Date
    df_full['Year'] = df_full['Date'].dt.year

    # Define input and target columns
    full_columns = df_full.columns
    input_cols = [full_columns[0]] + list(full_columns[2:-2]) + [full_columns[-1]]
    target_col = 'AQI'

    # Train-test split
    train_and_val_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_and_val_df, test_size=0.2, random_state=42)

    # Define inputs and target for training, validation, and test sets
    train_inputs = train_df[input_cols].copy()
    train_target = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_target = val_df[target_col].copy()
    test_inputs = test_df[input_cols].copy()
    test_target = test_df[target_col].copy()

    numerical_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df_full[numerical_cols])

    train_inputs[numerical_cols] = imputer.transform(train_inputs[numerical_cols])
    val_inputs[numerical_cols] = imputer.transform(val_inputs[numerical_cols])
    test_inputs[numerical_cols] = imputer.transform(test_inputs[numerical_cols])

    # Scale numerical columns
    scaler = StandardScaler().fit(train_inputs[numerical_cols])

    train_inputs[numerical_cols] = scaler.transform(train_inputs[numerical_cols])
    val_inputs[numerical_cols] = scaler.transform(val_inputs[numerical_cols])
    test_inputs[numerical_cols] = scaler.transform(test_inputs[numerical_cols])

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(df_full[categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

    # Define training, validation, and test data
    X_train = train_inputs[numerical_cols + encoded_cols]
    X_val = val_inputs[numerical_cols + encoded_cols]
    X_test = test_inputs[numerical_cols + encoded_cols]

    # Define a Decision Tree model and fit it on the training data
    tree = DecisionTreeRegressor(random_state=42)

    def try_model(model):
        model.fit(X_train, train_target)
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        train_r2_score = r2_score(train_target, train_preds)
        val_r2_score = r2_score(val_target, val_preds)
        test_r2_score = r2_score(test_target, test_preds)

        logging.info(f"Train R² Score: {train_r2_score}")
        logging.info(f"Validation R² Score: {val_r2_score}")
        logging.info(f"Test R² Score: {test_r2_score}")
        
        # Log metrics to Azure ML
        run.log("Train R² Score", train_r2_score)
        run.log("Validation R² Score", val_r2_score)
        run.log("Test R² Score", test_r2_score)

    try_model(tree)

    try:
        os.makedirs('./outputs', exist_ok=True)

        #Save model and preprocessors to outputs for Azure ML
        joblib.dump(tree, './outputs/dectree_aqi_model.pkl')

        model_path = 'outputs/dectree_aqi_model.pkl'
        Model.register(workspace=ws, model_path=model_path, model_name='dectree_aqi_model')
        logging.info("Model registered successfully")

    except Exception as e:
        logging.error(f"Error in model registration: {e}")
        raise

# Modify the end of train.py:
if __name__ == "__main__":
    # Hardcoded data path
    data_path = (Datastore.get(ws, 'aqi_pred_datastore'), 'city_day.csv')
    
    try:
        run_context = Run.get_context()
        # Check if running in Azure ML
        if isinstance(run_context, Run):
            # Running on Azure ML
            main(data_path)
        else:
            # Local execution - submit to Azure ML
            config = ScriptRunConfig(
                source_directory='.',
                script='train.py',
                compute_target=compute_target,
                environment=env
            )
            run = experiment.submit(config)
            run.wait_for_completion(show_output=True)
    except Exception as e:
        logging.error(f"Error in run context: {e}")
        # Fallback to local execution
        main(data_path)