import logging
import joblib
from utils import (
    load_data,
    encode_categorical_features,
    evaluate_model,
    plot_feature_importance,
    shap_analysis
)
from xgboost import XGBRegressor, DMatrix, cv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Configurable paths
    DATA_DIR = os.path.join(os.getcwd(), "data")
    MODEL_DIR = os.path.join(os.getcwd(), "models")
    PLOTS_DIR = os.path.join(os.getcwd(), "plots")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "X_train.csv")
    val_path = os.path.join(DATA_DIR, "X_val.csv")
    y_train_path = os.path.join(DATA_DIR, "y_train.csv")
    y_val_path = os.path.join(DATA_DIR, "y_val.csv")
    model_save_path = os.path.join(MODEL_DIR, "final_xgb_model.pkl")

    # Load datasets
    logging.info("Loading datasets...")
    X_train, X_val, y_train, y_val = load_data(train_path, val_path, y_train_path, y_val_path)

    # Encode categorical features
    logging.info("Encoding categorical features...")
    X_train, X_val = encode_categorical_features(X_train, X_val)

    # Convert training data to DMatrix format for XGBoost
    dtrain = DMatrix(X_train, label=y_train)

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'objective': ['reg:squarederror']
    }

    # Perform manual grid search using XGBoost's built-in CV
    logging.info("Performing grid search...")
    best_rmse = float("Inf")
    best_params = None

    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': learning_rate,
                'max_depth': max_depth,
            }
            results = cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=200,
                nfold=3,
                metrics='rmse',
                seed=42
            )
            mean_rmse = results['test-rmse-mean'].min()
            logging.info(f"Params: {params}, RMSE: {mean_rmse}")
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = params

    logging.info(f"Best Parameters: {best_params}")
    logging.info(f"Best RMSE from CV: {best_rmse}")

    # Train the model with the best parameters
    logging.info("Training the final model...")
    final_model = XGBRegressor(
        **best_params,
        n_estimators=200,
        random_state=42
    )
    final_model.fit(X_train, y_train)

    # Save the trained model using joblib
    logging.info(f"Saving the trained model to {model_save_path}...")
    joblib.dump(final_model, model_save_path)

    # Make predictions on the validation set
    logging.info("Making predictions...")
    y_pred = final_model.predict(X_val)

    # Evaluate the model
    logging.info("Evaluating the model...")
    rmse, mae = evaluate_model(y_val, y_pred)

    # Plot and save feature importance
    logging.info("Plotting feature importance...")
    feature_importance_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plot_feature_importance(final_model, X_train, save_path=feature_importance_path)

    # Perform and save SHAP analysis
    logging.info("Performing SHAP analysis...")
    summary_path = os.path.join(PLOTS_DIR, "shap_summary_plot.png")
    force_path = os.path.join(PLOTS_DIR, "shap_force_plot.html")
    shap_analysis(final_model, X_val, summary_path=summary_path, force_path=force_path)

if __name__ == "__main__":
    main()
