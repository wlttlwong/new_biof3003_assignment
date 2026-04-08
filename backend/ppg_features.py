import numpy as np
from scipy.stats import entropy

def extract_ppg_features(signal):
    """Extract 9 features from a PPG segment. Updated to match Scaler requirements."""
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 2:
        return np.zeros(9)  # Updated to 9
        
    mean = np.mean(arr)
    std = np.std(arr)
    std = max(std, 1e-7)
        
    diff = arr - mean
    
    # 1-4: Basic Stats
    skewness = np.mean(np.power(diff, 3)) / (np.power(std, 3) + 1e-7)
    kurtosis = np.mean(np.power(diff, 4)) / (np.power(std, 4) + 1e-7)
    
    # 5-8: Signal Characteristics
    signal_range = np.max(arr) - np.min(arr)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(diff)))) // 2
    rms = np.sqrt(np.mean(np.square(arr)))
    peak_to_peak = signal_range
    
    # 9: Shannon Entropy (The likely missing 9th feature)
    # This measures the complexity/noise of the signal
    hist, _ = np.histogram(arr, bins=10, density=True)
    hist = hist + 1e-9 # Avoid log(0)
    entropy_feat = -np.sum(hist * np.log2(hist))

    return np.array([
        mean, std, skewness, kurtosis,
        signal_range, zero_crossings,
        rms, peak_to_peak, entropy_feat
    ], dtype=float)