"""Extract 8 features from a PPG segment. Same feature set as course notebook (good/bad only)."""
import numpy as np


def extract_ppg_features(signal):
    """Extract 8 features from a PPG segment (list or array)."""
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 2:
        return np.zeros(8, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-7:
        std = 1e-7
    diff = arr - mean
    skewness = np.mean(np.power(diff, 3)) / (np.power(std, 3) + 1e-7)
    kurtosis = np.mean(np.power(diff, 4)) / (np.power(std, 4) + 1e-7)
    signal_range = np.max(arr) - np.min(arr)
    # Zero crossings: count sign changes (match notebook signbit-based definition)
    zero_crossings = int(np.sum(np.diff(np.signbit(arr).astype(int)) != 0))
    rms = np.sqrt(np.mean(np.square(arr)))
    peak_to_peak = signal_range
    return np.array(
        [
            mean,
            std,
            skewness,
            kurtosis,
            signal_range,
            zero_crossings,
            rms,
            peak_to_peak,
        ],
        dtype=float,
    )
