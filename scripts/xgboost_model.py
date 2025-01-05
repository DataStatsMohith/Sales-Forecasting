import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import joblib
import logging

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def load_data(data_dir):
    logging.info("Loading datasets...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv"))
    return X_train, X_val, y_train, y_val

def preprocess_data(X_train, X_val):
    logging.info("Preprocessing datasets...")
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=categorical_columns, drop_first=True)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_val

def train_model(X_train, y_train):
    logging.info("Training the XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    logging.info(f"Evaluation Metrics: RMSE = {rmse}, MAE = {mae}")
    return rmse, mae

def save_model(model, output_dir):
    model_path = os.path.join(output_dir, "final_xgb_model.pkl")
    logging.info(f"Saving the trained model to {model_path}...")
    joblib.dump(model, model_path)

def main():
    # Paths
    DATA_DIR = os.path.join(os.getcwd(), "data")
    OUTPUT_DIR = os.path.join(os.getcwd(), "models")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_data(DATA_DIR)
    X_train, X_val = preprocess_data(X_train, X_val)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_val, y_val)

    # Save model
    save_model(model, OUTPUT_DIR)

if __name__ == "__main__":
    main()
