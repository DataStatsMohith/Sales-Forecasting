import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_explore_data(file_path, dataset_name):
    """
    Load and explore a dataset.
    
    Parameters:
    - file_path (str): Path to the dataset.
    - dataset_name (str): Name of the dataset for logging purposes.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    try:
        logging.info(f"Loading {dataset_name} from {file_path}...")
        data = pd.read_csv(file_path)

        # Basic exploration
        logging.info(f"Displaying information for {dataset_name}:")
        logging.info(data.info())
        logging.info(f"First 5 rows of {dataset_name}:\n{data.head()}")

        # Check for missing values
        missing_values = data.isnull().sum()
        logging.info(f"Missing Values in {dataset_name}:\n{missing_values}\n")

        return data
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}. Please check the path.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while processing {dataset_name}: {e}")
        raise

def main():
    # Paths to datasets
    DATA_DIR = os.path.join(os.getcwd(), "data")
    store_path = os.path.join(DATA_DIR, "store.csv")
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    # Load and explore datasets
    store_data = load_and_explore_data(store_path, "Store Data")
    train_data = load_and_explore_data(train_path, "Train Data")
    test_data = load_and_explore_data(test_path, "Test Data")

if __name__ == "__main__":
    main()
