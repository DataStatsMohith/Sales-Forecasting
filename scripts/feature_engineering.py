import pandas as pd
import os

def add_lag_features(data, target_col, lags):
    for lag in lags:
        data[f"{target_col}_Lag_{lag}"] = data[target_col].shift(lag)
    return data

def add_rolling_features(data, target_col, windows):
    for window in windows:
        data[f"{target_col}_Rolling_Mean_{window}"] = data[target_col].rolling(window=window).mean()
    return data

def add_interaction_features(data):
    data['Promo_DayOfWeek'] = data['Promo'] * data['DayOfWeek']
    data['Promo_Customers'] = data['Promo'] * data['Customers']
    return data

def add_holiday_features(data, holiday_dates):
    data['Days_To_Holiday'] = data['Date'].apply(
        lambda x: min(abs((x - holiday).days) for holiday in holiday_dates)
    )
    return data

def main():
    # Paths
    DATA_DIR = os.path.join(os.getcwd(), "data")
    INPUT_PATH = os.path.join(DATA_DIR, "train_processed.csv")
    OUTPUT_PATH = os.path.join(DATA_DIR, "processed", "train_enhanced.csv")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load dataset
    train_data = pd.read_csv(INPUT_PATH, dtype={'StoreType': 'str'})
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.sort_values('Date', inplace=True)

    # Add features
    train_data = add_lag_features(train_data, "Sales", lags=[1, 7, 30])
    train_data = add_rolling_features(train_data, "Sales", windows=[7, 30])
    train_data = add_interaction_features(train_data)

    # Add weekend indicator
    train_data['IsWeekend'] = train_data['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)

    # Add holiday proximity
    holiday_dates = [
        pd.Timestamp("2023-12-25"),  # Example: Christmas
        pd.Timestamp("2024-01-01")  # Example: New Year
    ]
    train_data = add_holiday_features(train_data, holiday_dates)

    # Drop rows with NaN values
    train_data.dropna(inplace=True)

    # Save the enhanced dataset
    train_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature engineering complete. Enhanced dataset saved to {OUTPUT_PATH}.")

if __name__ == "__main__":
    main()
