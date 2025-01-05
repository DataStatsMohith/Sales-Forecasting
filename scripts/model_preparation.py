import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def split_and_save_data(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Splits the data into training and validation sets and saves the splits to files.

    Parameters:
    - input_path (str): Path to the input CSV file.
    - output_dir (str): Directory where the splits will be saved.
    - test_size (float): Proportion of the data to use as the validation set.
    - random_state (int): Seed for reproducibility.
    """
    try:
        # Load the dataset
        logging.info(f"Loading dataset from {input_path}...")
        data = pd.read_csv(input_path)

        # Ensure the necessary columns are present
        if "Sales" not in data.columns or "Date" not in data.columns:
            raise ValueError("The dataset must contain 'Sales' and 'Date' columns.")

        # Features and target
        X = data.drop(columns=["Sales", "Date"])  # Drop target and Date column
        y = data["Sales"]

        # Split the data
        logging.info("Splitting the data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the splits
        logging.info(f"Saving splits to {output_dir}...")
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)

        logging.info("Data split complete. Files saved as 'X_train.csv', 'X_val.csv', 'y_train.csv', 'y_val.csv'.")
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_path}. Please check the path and try again.")
    except Exception as e:
        logging.error(f"An error occurred during data splitting: {e}")

if __name__ == "__main__":
    # Paths
    INPUT_PATH = "data/train_engineered.csv"
    OUTPUT_DIR = "data"

    # Split and save the data
    split_and_save_data(INPUT_PATH, OUTPUT_DIR)
