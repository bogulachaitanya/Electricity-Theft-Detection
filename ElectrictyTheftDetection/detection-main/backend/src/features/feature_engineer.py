import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.fftpack import fft
import argparse
import os

class FeatureEngineer:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.meta_cols = None
        self.date_cols = None
        self.features_df = pd.DataFrame()

    def load_data(self):
        print(f"Loading preprocessed dataset from {self.input_file}...")
        self.df = pd.read_csv(self.input_file)
        self.meta_cols = [c for c in self.df.columns if c in ['CONS_NO', 'FLAG']]
        self.date_cols = [c for c in self.df.columns if c not in self.meta_cols]
        # Use values array for fast computation
        self.ts_matrix = self.df[self.date_cols].values
        self.features_df[self.meta_cols] = self.df[self.meta_cols]

    def compute_statistical_features(self):
        print("Computing Time-Domain Statistical Features...")
        self.features_df['mean_consumption'] = np.mean(self.ts_matrix, axis=1)
        self.features_df['std_consumption'] = np.std(self.ts_matrix, axis=1)
        self.features_df['max_consumption'] = np.max(self.ts_matrix, axis=1)
        self.features_df['min_consumption'] = np.min(self.ts_matrix, axis=1)
        self.features_df['median_consumption'] = np.median(self.ts_matrix, axis=1)
        
        # Coefficient of variation (normalized variability)
        self.features_df['cv_consumption'] = self.features_df['std_consumption'] / (self.features_df['mean_consumption'] + 1e-6)
        
        # Interquartile range (robust spread measure)
        q75 = np.percentile(self.ts_matrix, 75, axis=1)
        q25 = np.percentile(self.ts_matrix, 25, axis=1)
        self.features_df['iqr_consumption'] = q75 - q25
        
        # Skewness and Kurtosis (distribution shape — theft distorts these)
        self.features_df['skewness'] = stats.skew(self.ts_matrix, axis=1)
        self.features_df['kurtosis'] = stats.kurtosis(self.ts_matrix, axis=1)
        
        # Gradients (day-over-day leaps)
        gradient = np.diff(self.ts_matrix, axis=1)
        self.features_df['mean_gradient'] = np.mean(gradient, axis=1)
        self.features_df['std_gradient'] = np.std(gradient, axis=1)
        self.features_df['max_gradient'] = np.max(gradient, axis=1)
        self.features_df['min_gradient'] = np.min(gradient, axis=1)
        
        # Rolling averages (using last 7 and 30 days as static features)
        self.features_df['last_7d_mean'] = np.mean(self.ts_matrix[:, -7:], axis=1)
        self.features_df['last_30d_mean'] = np.mean(self.ts_matrix[:, -30:], axis=1)
        self.features_df['mean_delta_7_30'] = self.features_df['last_7d_mean'] - self.features_df['last_30d_mean']
        
        # First half vs second half (trend detection)
        mid = self.ts_matrix.shape[1] // 2
        first_half_mean = np.mean(self.ts_matrix[:, :mid], axis=1)
        second_half_mean = np.mean(self.ts_matrix[:, mid:], axis=1)
        self.features_df['half_period_ratio'] = second_half_mean / (first_half_mean + 1e-6)
        
    def compute_pattern_features(self):
        print("Computing Consumption Pattern Features...")
        peak = self.features_df['max_consumption']
        base = self.features_df['min_consumption']
        
        self.features_df['peak_to_base_ratio'] = peak / (base + 1e-6)
        
        # Range (max - min)
        self.features_df['range_consumption'] = peak - base
        
        # Entropy
        entropies = []
        for row in self.ts_matrix:
            hist, _ = np.histogram(row, bins=20, density=True)
            entropies.append(stats.entropy(hist + 1e-6))
        self.features_df['shannon_entropy'] = entropies

        # Weekday/Weekend split
        date_series = pd.to_datetime(self.date_cols, errors='coerce')
        is_weekend = date_series.dayofweek >= 5
        
        weekend_data = self.ts_matrix[:, is_weekend]
        weekday_data = self.ts_matrix[:, ~is_weekend]
        
        self.features_df['weekend_mean'] = np.mean(weekend_data, axis=1) if weekend_data.shape[1] > 0 else 0
        self.features_df['weekday_mean'] = np.mean(weekday_data, axis=1) if weekday_data.shape[1] > 0 else 0
        self.features_df['weekday_weekend_ratio'] = self.features_df['weekday_mean'] / (self.features_df['weekend_mean'] + 1e-6)

        # Weekend std vs weekday std (theft patterns differ on weekends)
        self.features_df['weekend_std'] = np.std(weekend_data, axis=1) if weekend_data.shape[1] > 0 else 0
        self.features_df['weekday_std'] = np.std(weekday_data, axis=1) if weekday_data.shape[1] > 0 else 0

        # Pragmatic approach: last 5 days of the entire provided time series
        self.features_df['last_5d_vs_prev_ratio'] = np.mean(self.ts_matrix[:, -5:], axis=1) / (np.mean(self.ts_matrix[:, -30:-5], axis=1) + 1e-6)

        # Autocorrelation at lag 1 and lag 7 (weekly pattern regularity)
        print("Computing Autocorrelation Features...")
        autocorr_lag1 = []
        autocorr_lag7 = []
        for row in self.ts_matrix:
            if len(row) > 7:
                ac1 = np.corrcoef(row[:-1], row[1:])[0, 1] if np.std(row) > 1e-6 else 0
                ac7 = np.corrcoef(row[:-7], row[7:])[0, 1] if np.std(row) > 1e-6 else 0
                autocorr_lag1.append(ac1 if not np.isnan(ac1) else 0)
                autocorr_lag7.append(ac7 if not np.isnan(ac7) else 0)
            else:
                autocorr_lag1.append(0)
                autocorr_lag7.append(0)
        self.features_df['autocorr_lag1'] = autocorr_lag1
        self.features_df['autocorr_lag7'] = autocorr_lag7

    def compute_anomaly_indicators(self):
        print("Computing Anomaly Indicator Features...")
        # Zero readings (Direct theft indicators)
        self.features_df['zero_reading_count'] = np.sum(self.ts_matrix == 0, axis=1)
        self.features_df['zero_reading_ratio'] = self.features_df['zero_reading_count'] / self.ts_matrix.shape[1]
        
        consecutive_zeros = []
        for row in (self.ts_matrix == 0):
            diffs = np.diff(np.concatenate(([0], row, [0])))
            lengths = np.where(diffs == -1)[0] - np.where(diffs == 1)[0]
            consecutive_zeros.append(np.max(lengths) if len(lengths) > 0 else 0)
        self.features_df['max_zero_streak'] = consecutive_zeros

        # Low reading persistency
        mean_val = self.features_df['mean_consumption'].values.reshape(-1, 1)
        is_very_low = self.ts_matrix < (0.2 * mean_val)
        self.features_df['low_reading_persistency'] = np.sum(is_very_low, axis=1)
        self.features_df['low_reading_ratio'] = self.features_df['low_reading_persistency'] / self.ts_matrix.shape[1]
        
        # Sudden drops (> 2 std devs)
        gradient = np.diff(self.ts_matrix, axis=1)
        std_val = self.features_df['std_consumption'].values.reshape(-1, 1)
        self.features_df['sudden_drop_count'] = np.sum(gradient < (-2.0 * (std_val + 1e-6)), axis=1)
        
        # Sudden spikes (> 2 std devs positive)
        self.features_df['sudden_spike_count'] = np.sum(gradient > (2.0 * (std_val + 1e-6)), axis=1)
        
        # Spike then drop (signature of bypass installation)
        spike_then_drop = []
        for row_grad in gradient:
            triggered = 0
            for i in range(len(row_grad)-1):
                if row_grad[i] > 2.0 and row_grad[i+1] < -2.0:
                    triggered += 1
            spike_then_drop.append(triggered)
        self.features_df['spike_then_drop_count'] = spike_then_drop

        # Benford's Law deviation (digit distribution anomaly)
        print("Computing Benford's Law Deviation...")
        benford_expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
        benford_deviations = []
        for row in self.ts_matrix:
            positive_vals = row[row > 0]
            if len(positive_vals) > 10:
                first_digits = np.array([int(str(abs(v)).replace('.', '').replace('-', '').lstrip('0')[0]) 
                                        for v in positive_vals if v != 0 and str(abs(v)).replace('.', '').replace('-', '').lstrip('0')])
                first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]
                if len(first_digits) > 5:
                    observed = np.array([(first_digits == d).sum() / len(first_digits) for d in range(1, 10)])
                    deviation = np.sum((observed - benford_expected) ** 2)
                    benford_deviations.append(deviation)
                else:
                    benford_deviations.append(0)
            else:
                benford_deviations.append(0)
        self.features_df['benford_deviation'] = benford_deviations

    def compute_fourier_features(self):
        print("Computing Fourier Frequency Features...")
        n = self.ts_matrix.shape[1]
        fft_result = np.abs(fft(self.ts_matrix, axis=1))
        
        freqs = np.fft.fftfreq(n)
        pos = freqs > 0
        fft_pos = fft_result[:, pos]
        freqs_pos = freqs[pos]
        
        self.features_df['dominant_freq'] = freqs_pos[np.argmax(fft_pos, axis=1)]
        self.features_df['freq_amplitude'] = np.max(fft_pos, axis=1)
        
        # Total spectral energy
        self.features_df['spectral_energy'] = np.sum(fft_pos ** 2, axis=1)
        
        # Spectral centroid (center of mass of spectrum)
        total_amp = np.sum(fft_pos, axis=1, keepdims=True)
        self.features_df['spectral_centroid'] = np.sum(fft_pos * freqs_pos, axis=1) / (total_amp.flatten() + 1e-6)
        
        # Specific bands
        self.features_df['weekly_vibe'] = fft_pos[:, np.argmin(np.abs(freqs_pos - 1/7.0))]
        self.features_df['monthly_vibe'] = fft_pos[:, np.argmin(np.abs(freqs_pos - 1/30.0))]
        
        # Spectral flatness (how noise-like vs tonal the signal is)
        log_mean = np.mean(np.log(fft_pos + 1e-10), axis=1)
        arith_mean = np.mean(fft_pos, axis=1)
        self.features_df['spectral_flatness'] = np.exp(log_mean) / (arith_mean + 1e-6)

    def compute_advanced_features(self):
        """NEW: Additional advanced features for better discrimination."""
        print("Computing Advanced Theft-Detection Features...")
        
        # Rolling window volatility (30-day windows)
        n_cols = self.ts_matrix.shape[1]
        window = 30
        if n_cols >= window * 2:
            rolling_stds = []
            for start in range(0, n_cols - window, window):
                chunk = self.ts_matrix[:, start:start+window]
                rolling_stds.append(np.std(chunk, axis=1))
            rolling_stds = np.array(rolling_stds)  # shape: (n_windows, n_samples)
            
            # Volatility of volatility (meta-volatility indicates tampering)
            self.features_df['volatility_of_volatility'] = np.std(rolling_stds, axis=0)
            
            # Max volatility change between consecutive windows
            vol_changes = np.diff(rolling_stds, axis=0)
            self.features_df['max_volatility_change'] = np.max(np.abs(vol_changes), axis=0) if vol_changes.shape[0] > 0 else 0

        # Consumption below neighborhood percentile (relative anomaly)
        global_median = np.median(self.features_df['mean_consumption'])
        self.features_df['below_global_median'] = (self.features_df['mean_consumption'] < global_median * 0.5).astype(int)
        
        # Night-day ratio approximation (theft often visible in usage patterns)
        # Use first third vs last two thirds of daily data as proxy
        third = n_cols // 3
        if third > 0:
            first_third = np.mean(self.ts_matrix[:, :third], axis=1)
            rest = np.mean(self.ts_matrix[:, third:], axis=1)
            self.features_df['period_ratio'] = first_third / (rest + 1e-6)

    def save_features(self):
        n_features = len(self.features_df.columns) - len([c for c in self.features_df.columns if c in ['CONS_NO', 'FLAG']])
        print(f"Saving {n_features} dimension feature matrix to {self.output_file}...")
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.features_df.to_csv(self.output_file, index=False)
        try:
            self.features_df.to_parquet(self.output_file.replace('.csv', '.parquet'), index=False)
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/clean_sgcc.csv")
    parser.add_argument("--output", default="data/processed/features_sgcc.csv")
    args = parser.parse_args()
    
    fe = FeatureEngineer(args.input, args.output)
    fe.load_data()
    fe.compute_statistical_features()
    fe.compute_pattern_features()
    fe.compute_anomaly_indicators()
    fe.compute_fourier_features()
    fe.compute_advanced_features()  # NEW
    fe.save_features()
    print("Feature Engineering complete!")
