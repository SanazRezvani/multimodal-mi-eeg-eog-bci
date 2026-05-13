import numpy as np
from scipy.signal import welch
from mne.decoding import CSP


def compute_bandpower(epoch_data, sfreq, band):
    low, high = band

    freqs, psd = welch(
        epoch_data,
        fs=sfreq,
        nperseg=int(sfreq),
        axis=-1,
    )

    band_mask = (freqs >= low) & (freqs <= high)

    return psd[:, band_mask].mean(axis=1)


def extract_epoch_features(epochs, motor_channels, sfreq):
    data = epochs.copy().pick(motor_channels).get_data()
    # Shape: epochs x channels x samples

    features = []

    for epoch in data:
        mu_power = compute_bandpower(epoch, sfreq, band=(8, 13))
        beta_power = compute_bandpower(epoch, sfreq, band=(13, 30))

        epoch_features = np.concatenate([mu_power, beta_power])
        features.append(epoch_features)

    return np.array(features)


def label_epochs_with_blinks(df, events, tmin, tmax, sfreq):
    blink_labels = []

    n_samples_window = int((tmax - tmin) * sfreq)

    for event in events:
        start_sample = event[0]
        end_sample = start_sample + n_samples_window

        epoch_blinks = df["Blinks"].iloc[start_sample:end_sample]

        has_blink = int(epoch_blinks.sum() > 0)
        blink_labels.append(has_blink)

    return np.array(blink_labels)



def extract_csp_features(epochs, decoding_channels, n_components=2):
    """
    Extract CSP features from motor imagery epochs.

    Returns:
        X_csp: CSP feature matrix
        y: class labels
        csp: fitted CSP object
    """

    epochs_motor = epochs.copy().pick(decoding_channels)

    X = epochs_motor.get_data()
    y = epochs_motor.events[:, -1]

    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False
    )

    X_csp = csp.fit_transform(X, y)

    return X_csp, y, csp