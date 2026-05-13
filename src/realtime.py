import time
import numpy as np
import pandas as pd


def extract_sliding_window_data(
    epochs,
    sfreq,
    window_length_sec=1.0,
    step_size_sec=0.25,
):
    """
    Convert epoched EEG data into sliding-window samples.

    Returns:
        X_windows: shape = windows x channels x samples
        y_windows: labels for each window
    """

    data = epochs.get_data()
    labels = epochs.events[:, -1]

    window_samples = int(window_length_sec * sfreq)
    step_samples = int(step_size_sec * sfreq)

    X_windows = []
    y_windows = []

    for epoch_idx, epoch_data in enumerate(data):
        true_label = labels[epoch_idx]
        n_samples = epoch_data.shape[1]

        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples

            window_data = epoch_data[:, start:end]

            X_windows.append(window_data)
            y_windows.append(true_label)

    return np.array(X_windows), np.array(y_windows)


def simulate_realtime_decoding_csp(
    epochs,
    classifier_model,
    csp_model,
    decoding_channels,
    sfreq,
    window_length_sec=1.0,
    step_size_sec=0.25,
):
    """
    Simulate real-time decoding using CSP features.

    Each sliding window is transformed using the fitted CSP model,
    then classified using the trained classifier.
    """

    epochs_decoding = epochs.copy().pick(decoding_channels)

    data = epochs_decoding.get_data()
    labels = epochs_decoding.events[:, -1]

    window_samples = int(window_length_sec * sfreq)
    step_samples = int(step_size_sec * sfreq)

    prediction_rows = []

    for epoch_idx, epoch_data in enumerate(data):
        true_label = labels[epoch_idx]
        n_samples = epoch_data.shape[1]

        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window_data = epoch_data[:, start:end]

            tic = time.perf_counter()

            # CSP expects shape: trials x channels x samples
            window_csp_features = csp_model.transform(
                window_data[np.newaxis, :, :]
            )

            predicted_label = classifier_model.predict(
                window_csp_features
            )[0]

            latency_ms = (time.perf_counter() - tic) * 1000

            prediction_rows.append({
                "epoch_index": epoch_idx,
                "window_start_sec": start / sfreq,
                "window_end_sec": end / sfreq,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "latency_ms": latency_ms,
            })

    return pd.DataFrame(prediction_rows)