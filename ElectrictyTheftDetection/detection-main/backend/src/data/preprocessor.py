import pandas as pd
import numpy as np
import os

class PreprocessingPipeline:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        
    def load_csv(self):
        print(f"Loading raw dataset from {self.input_file}...")
        self.df = pd.read_csv(self.input_file)
        
        cols = self.df.columns.tolist()
        if 'FLAG' in cols and 'CONS_NO' in cols:
            self.date_cols = [c for c in cols if c not in ['FLAG', 'CONS_NO']]
            self.meta_cols = ['CONS_NO', 'FLAG']
        elif 'FLAG' in cols:
            # Maybe some variations
            self.date_cols = [c for c in cols if c != 'FLAG']
            self.meta_cols = ['FLAG']
        else:
            raise ValueError("Required columns (FLAG, CONS_NO) not found.")
            
        self.ts_data = self.df[self.date_cols].copy()
        
        # Convert all to numeric, coercing errors to NaN
        print("Converting data to numeric...")
        self.ts_data = self.ts_data.apply(pd.to_numeric, errors='coerce')
        
    def clip_outliers(self):
        print("Clipping outliers (negative values and high spikes)...")
        # Set negative values to 0
        self.ts_data[self.ts_data < 0] = 0
        
        # Clip upper limit: Household mean + 3 * standard deviation
        household_means = self.ts_data.mean(axis=1)
        household_stds = self.ts_data.std(axis=1)
        upper_bounds = household_means + 3 * household_stds
        
        # Clip column by column (Pandas convention)
        self.ts_data = self.ts_data.clip(upper=upper_bounds, axis=0)
        print("Outliers clipped successfully.")

    def validate_schema(self):
        print("Validating schema...")
        if not self.meta_cols or not self.date_cols:
            raise ValueError("Schema validation failed: Missing required columns.")
        # Data type checks already incorporated in load_csv via pd.to_numeric
        print("Schema validation passed.")

    def resample_to_daily(self):
        print("Resampling to daily frequency...")
        # The SGCC dataset is already in daily format per column names like '1/1/2014'.
        # This function acts as a placeholder conforming to the architectural blueprint.
        pass


    def impute_gaps(self):
        print("Imputing missing values (forward fill up to 3 days, then remaining to 0)...")
        # Forward fill up to 3 missing days
        self.ts_data = self.ts_data.ffill(axis=1, limit=3)
        # Fill remaining NaNs with 0
        self.ts_data = self.ts_data.fillna(0)
        
    def apply_kalman_filter(self, transition_covariance=0.01, observation_covariance=0.1):
        """Simple 1D Kalman filter for smoothing sensor noise."""
        print("Applying Kalman Filtering to smooth sensor noise...")
        
        def kalman_smooth(vals):
            # A simple implementation to avoid large dependencies if possible
            # But since we have scipy, we can use it or just the 1D recursive equations
            n = len(vals)
            if n == 0: return vals
            
            x_hat = np.zeros(n)
            p = np.zeros(n)
            x_hat_minus = np.zeros(n)
            p_minus = np.zeros(n)
            k = np.zeros(n)
            
            # Initial guesses
            x_hat[0] = vals[0] if not np.isnan(vals[0]) else 0
            p[0] = 1.0
            
            for i in range(1, n):
                # Prediction
                x_hat_minus[i] = x_hat[i-1]
                p_minus[i] = p[i-1] + transition_covariance
                
                # Measurement
                v = vals[i] if not np.isnan(vals[i]) else x_hat_minus[i]
                
                # Update
                k[i] = p_minus[i] / (p_minus[i] + observation_covariance)
                x_hat[i] = x_hat_minus[i] + k[i] * (v - x_hat_minus[i])
                p[i] = (1 - k[i]) * p_minus[i]
                
            return x_hat

        # Apply to each household (row)
        self.ts_data = self.ts_data.apply(lambda row: kalman_smooth(row.values), axis=1, result_type='broadcast')
        print("Kalman smoothing completed.")

    def apply_stl_decomposition(self, period=7):
        """
        Seasonal-Trend decomposition using LOESS.
        Extracts Trend and Residuals which are robust features for theft detection.
        """
        print(f"Applying STL Decomposition (Period={period})...")
        from statsmodels.tsa.seasonal import STL
        
        # We'll replace the raw data with the Trend + Residual for a cleaner signal,
        # or just keep the original data but smoothed.
        def get_stl_trend(vals):
            try:
                # Need at least 2 periods for STL
                if len(vals) < 2 * period:
                    return vals
                res = STL(vals, period=period, robust=True).fit()
                return res.trend + res.resid  # This removes the repeating 'seasonal' noise
            except Exception:
                return vals

        self.ts_data = self.ts_data.apply(lambda row: get_stl_trend(row.values), axis=1, result_type='broadcast')
        print("STL Decomposition completed.")

    def normalize_per_household(self):
        print("Normalizing consumption values (Z-score normalization)...")
        household_means = self.ts_data.mean(axis=1)
        household_stds = self.ts_data.std(axis=1)
        
        # Avoid division by zero for households with constant 0 consumption
        household_stds[household_stds == 0] = 1 
        
        self.ts_data = self.ts_data.subtract(household_means, axis=0).divide(household_stds, axis=0)

    def save_processed_data(self):
        print(f"Combining and saving processed data to {self.output_file}...")
        df_clean = self.ts_data.copy()
        for col in self.meta_cols:
            df_clean[col] = self.df[col]
            
        # Reorder to keep metadata at the end
        df_clean = df_clean[self.date_cols + self.meta_cols]
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        df_clean.to_csv(self.output_file, index=False)
        print("Data processing pipeline completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Electricity Theft Detection - Data Preprocessing')
    parser.add_argument('--input', type=str, default='data set.csv', help='Input raw CSV file')
    parser.add_argument('--output', type=str, default='data/processed/clean_dataset.csv', help='Output processed CSV file')
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline(input_file=args.input, output_file=args.output)
    pipeline.load_csv()
    pipeline.validate_schema()
    pipeline.clip_outliers()
    pipeline.impute_gaps()
    
    # NEW REFINED CLEANING STEPS
    pipeline.apply_kalman_filter()
    pipeline.apply_stl_decomposition()
    
    pipeline.resample_to_daily()
    pipeline.normalize_per_household()
    pipeline.save_processed_data()
