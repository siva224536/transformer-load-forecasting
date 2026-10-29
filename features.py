import numpy as np
from scipy.signal import welch

def extract_features(signal, fs=1000):
    """Extract time-domain and frequency-domain features from one signal window."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 16 or not np.all(np.isfinite(x)):
        raise ValueError("Signal must be a finite 1-D array with at least 16 samples.")
    centered = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x))
    mean_abs = float(np.mean(np.abs(x)))
    crest = float(peak / rms) if rms > 1e-12 else 0.0
    skew = float(np.mean(centered**3) / (np.std(centered)**3 + 1e-12))
    kurtosis = float(np.mean(centered**4) / (np.std(centered)**4 + 1e-12))
    zcr = float(np.mean(np.diff(np.signbit(centered)) != 0))
    freqs, psd = welch(centered, fs=fs, nperseg=min(256, len(centered)))
    total_power = float(np.sum(psd) + 1e-12)
    dominant_frequency = float(freqs[int(np.argmax(psd))])
    spectral_centroid = float(np.sum(freqs * psd) / total_power)
    band_mask = (freqs >= 100) & (freqs <= fs/2)
    high_frequency_ratio = float(np.sum(psd[band_mask]) / total_power)
    return {
        "mean": float(np.mean(x)), "rms_current": rms, "peak_current": peak,
        "standard_deviation": std, "mean_absolute_current": mean_abs,
        "crest_factor": crest, "skewness": skew, "kurtosis": kurtosis,
        "zero_crossing_rate": zcr, "dominant_frequency_hz": dominant_frequency,
        "spectral_centroid_hz": spectral_centroid,
        "high_frequency_power_ratio": high_frequency_ratio
    }
