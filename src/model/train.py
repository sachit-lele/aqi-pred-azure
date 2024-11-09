import numpy as np
import pandas as pd
import logging
import argparse
import datetime
from warnings import filterwarnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)
filterwarnings('ignore')

def main(data_path):
    # Load dataset
    df_city_day = pd.read_csv(data_path)

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
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
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
        return

    try_model(tree)

    # Save model and preprocessors to outputs for Azure ML
    joblib.dump(tree, './outputs/dectree_aqi_model.pkl')
    joblib.dump(imputer, './outputs/imputer.pkl')
    joblib.dump(scaler, './outputs/scaler.pkl')
    joblib.dump(encoder, './outputs/encoder.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the input CSV data file")
    args = parser.parse_args()
    main(args.data_path)
