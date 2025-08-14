# src/pipeline.py

import pandas as pd
from pathlib import Path

def run_pipeline():
    """
    Processes raw data and saves it in the processed data folder.
    """
    print("Starting data processing pipeline...")

    # Define paths
    base_path = Path(__file__).parent.parent
    raw_data_path = base_path / "data/raw/mock_loan_data.csv"
    processed_data_path = base_path / "data/processed/"
    
    # Create processed data directory if it doesn't exist
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} rows from raw data.")

    # --- Feature Engineering ---
    # Create a debt-to-income style ratio (example feature)
    df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']

    # One-hot encode the 'race' column for modeling
    df = pd.get_dummies(df, columns=['race'], prefix='race', drop_first=True)
    
    # Rename columns to be compatible with XGBoost
    df = df.rename(columns={"race_GroupB": "race_groupb"})

    print("Feature engineering complete.")
    print("Final columns:", df.columns.tolist())

    # Save processed data
    output_file = processed_data_path / "processed_loan_data.parquet"
    df.to_parquet(output_file)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    run_pipeline()