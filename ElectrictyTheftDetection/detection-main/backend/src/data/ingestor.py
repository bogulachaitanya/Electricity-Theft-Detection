import pandas as pd
import os

class DataIngestor:
    def __init__(self, raw_path):
        self.raw_path = raw_path

    def load_sgcc(self):
        """Loads the SGCC (State Grid Corporation of China) dataset."""
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_path}")
        
        print(f"Ingesting raw data from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)
        
        # Basic validation
        if 'CONS_NO' not in df.columns or 'FLAG' not in df.columns:
            raise ValueError("Dataset missing critical columns (CONS_NO or FLAG)")
            
        print(f"Successfully ingested {len(df)} rows.")
        return df

    def validate_schema(self, df):
        """Validates that the dates are consistently formatted as columns."""
        date_cols = [c for c in df.columns if c not in ['CONS_NO', 'FLAG']]
        try:
            pd.to_datetime(date_cols)
            print("Schema Validation: PASS (Date columns confirmed)")
            return True
        except:
            print("Schema Validation: WARNING (Non-date columns detected in time-series block)")
            return False
