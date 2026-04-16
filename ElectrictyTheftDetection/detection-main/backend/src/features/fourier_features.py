import numpy as np
from scipy.fft import fft, fftfreq

def extract_fourier_features(series, top_k=5):
    """Computes FFT and returns the top K dominant frequencies and their magnitudes."""
    if len(series) < 2:
        return [0] * (top_k * 2)
    
    # Remove mean to focus on oscillations
    signal = series - np.mean(series)
    N = len(signal)
    
    yf = fft(signal)
    xf = fftfreq(N, 1) # Frequency in cycles per day
    
    # Get absolute magnitudes
    mags = np.abs(yf[:N//2])
    freqs = xf[:N//2]
    
    # Sort by magnitude
    indices = np.argsort(mags)[-top_k:][::-True]
    
    features = []
    for idx in indices:
        features.append(freqs[idx])    # Frequency
        features.append(mags[idx])     # Strength
        
    # Pad if needed
    while len(features) < (top_k * 2):
        features.append(0)
        
    return features
