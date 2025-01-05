import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# General utility functions
def load_dataset(path, low_memory=True):
    """Load a dataset from the given path."""
    try:
        logging.info(f"Loading dataset from {path}...")
        return pd.read_csv(path, low_memory=low_memory)
    except FileNotFoundError:
        logging.error(f"File not found at {path}. Please check the path and try again.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        raise

def save_dataset(data, path):
    """Save a dataset to the given path."""
    try:
        logging.info(f"Saving dataset to {path}...")
        data.to_csv(path, index=False)
    except Exception as e:
        logging.error(f"An error occurred while saving the dataset: {e}")
        raise

def load_data(train_path, val_path, y_train_path, y_val_path):
    """Load training and validation datasets."""
    logging.info("Loading datasets...")
    X_train = load_dataset(train_path)
    X_val = load_dataset(val_path)
    y_train = load_dataset(y_train_path)
    y_val = load_dataset(y_val_path)
    return X_train, X_val, y_train, y_val

def encode_categorical_features(X_train, X_val):
    """One-hot encode categorical features and align columns."""
    try:
        logging.info("Encoding categorical features...")
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
        X_val = pd.get_dummies(X_val, columns=categorical_columns, drop_first=True)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        return X_train, X_val
    except Exception as e:
        logging.error(f"Error during encoding of categorical features: {e}")
        raise

def evaluate_model(y_val, y_pred):
    """Evaluate model performance."""
    try:
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
        logging.info(f"Mean Absolute Error (MAE): {mae}")
        return rmse, mae
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

# Visualization utilities
def plot_feature_importance(model, X_train, save_path="feature_importance.png"):
    """Plot and save feature importance."""
    try:
        logging.info("Plotting feature importance...")
        plt.figure(figsize=(12, 8))
        importance = model.feature_importances_
        features = X_train.columns
        sorted_idx = importance.argsort()

        plt.barh(features[sorted_idx], importance[sorted_idx], color="skyblue")
        plt.xlabel("Feature Importance", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.title("Feature Importance in XGBoost Model", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logging.info(f"Feature importance plot saved to {save_path}.")
        plt.show()
    except Exception as e:
        logging.error(f"Error while plotting feature importance: {e}")
        raise

def shap_analysis(model, X_val, summary_path="shap_summary_plot.png", force_path="shap_force_plot.html"):
    """Perform SHAP analysis and save plots."""
    try:
        logging.info("Performing SHAP analysis...")
        explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_val, check_additivity=False)

        # Save SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_val, show=False)
        plt.title("SHAP Summary Plot", fontsize=16)
        plt.savefig(summary_path, dpi=300)
        logging.info(f"SHAP summary plot saved to {summary_path}.")
        plt.show()

        # Save SHAP force plot for a single prediction
        sample_index = 0
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[sample_index],
            X_val.iloc[sample_index]
        )
        shap.save_html(force_path, force_plot)
        logging.info(f"SHAP force plot saved to {force_path}.")
    except Exception as e:
        logging.error(f"Error during SHAP analysis: {e}")
        raise
