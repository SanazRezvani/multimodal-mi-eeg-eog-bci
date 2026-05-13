import numpy as np


def extract_eog_features(epoch_data):
    """
    Extract simple features from EOG (HEO) channel.

    epoch_data shape:
        channels x samples

    Returns:
        feature vector (1D)
    """

    # Assume last channel is HEO (or pass index explicitly if needed)
    heo_signal = epoch_data[-1, :]

    variance = np.var(heo_signal)
    mean_abs = np.mean(np.abs(heo_signal))
    max_amp = np.max(np.abs(heo_signal))

    return np.array([variance, mean_abs, max_amp])