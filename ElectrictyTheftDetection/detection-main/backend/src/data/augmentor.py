import numpy as np
import pandas as pd

class TheftAugmentor:
    """Class to generate synthetic theft patterns for training enhancement."""
    
    @staticmethod
    def apply_partial_reduction(consumption_series, reduction_factor=0.5):
        """Simulates a partial bypass where consumption drops by a fixed percentage."""
        return consumption_series * (1 - reduction_factor)
    
    @staticmethod
    def apply_zero_streak(consumption_series, start_day, duration):
        """Simulates a meter tamper or total bypass for a fixed period."""
        new_series = consumption_series.copy()
        new_series[start_day:start_day+duration] = 0
        return new_series
    
    @staticmethod
    def apply_zigzag_bypass(consumption_series, reduction_factor=0.4):
        """Alternates between normal and reduced. Harder to detect."""
        mask = np.tile([1, 1 - reduction_factor], len(consumption_series) // 2 + 1)[:len(consumption_series)]
        return consumption_series * mask

    @staticmethod
    def apply_gradual_reduction(consumption_series, target_reduction=0.7):
        """Linearly decreases consumption over time."""
        weights = np.linspace(1.0, 1.0 - target_reduction, len(consumption_series))
        return consumption_series * weights

    @staticmethod
    def apply_irregular_reduction(consumption_series):
        """Simulates irregular tampering where some days are reduced and others are not."""
        mask = np.random.choice([0.2, 1.0], size=len(consumption_series), p=[0.4, 0.6])
        return consumption_series * mask
