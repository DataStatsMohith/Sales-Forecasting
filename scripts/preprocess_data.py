import logging
from utils import load_dataset, save_dataset
import pandas as pd
import os

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Configurable paths
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

store_path = os.path.join(DATA_DIR, "store.csv")
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train_output_path = os.path.join(OUTPUT_DIR, "train_processed.csv")
test_output_path = os.path.join(OUTPUT_DIR, "test_processed.csv")

try:
    # Load datasets
    logging.info("Loading datasets...")
    store_data = load_dataset(store_path)
    train_data = load_dataset(train_path, low_memory=False)
    test_data = load_dataset(test_path)

    # Handle missing values
    logging.info("Handling missing values...")
    store_data['CompetitionDistance'] = store_data['CompetitionDistance'].fillna(store_data['CompetitionDistance'].median())
    store_data.fillna({'CompetitionOpenSinceMonth': 0, 'CompetitionOpenSinceYear': 0,
                       'Promo2SinceWeek': 0, 'Promo2SinceYear': 0, 'PromoInterval': 'None'}, inplace=True)
    test_data['Open'] = test_data['Open'].fillna(1)

    # Convert Date to datetime format
    logging.info("Converting Date columns to datetime format...")
    for df in [train_data, test_data]:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Merge datasets
    logging.info("Merging datasets with store data...")
    train_data = train_data.merge(store_data, on='Store', how='left')
    test_data = test_data.merge(store_data, on='Store', how='left')

    # Save processed datasets
    logging.info("Saving processed datasets...")
    save_dataset(train_data, train_output_path)
    save_dataset(test_data, test_output_path)

    logging.info("Data preprocessing complete. Files saved in 'processed' directory.")
except Exception as e:
    logging.error(f"An error occurred during preprocessing: {e}")
