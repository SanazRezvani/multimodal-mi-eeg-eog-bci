import os
import numpy as np
from mne.decoding import CSP

from src.config import (
    CSV_PATH,
    RESULTS_DIR,
    SFREQ,
    EEG_CHANNELS,
    EOG_CHANNELS,
    DECODING_CHANNELS,
    EVENT_ID,
    TMIN,
    TMAX,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    RANDOM_STATE,
    WINDOW_LENGTH_SEC,
    STEP_SIZE_SEC,
    CSP_COMPONENTS,
)

from src.load_data import load_csv

from src.mne_processing import (
    create_raw_from_dataframe,
    filter_raw,
    create_events_from_cues,
    create_epochs,
)

from src.classification import train_evaluate_classifier

from src.realtime import (
    extract_sliding_window_data,
    simulate_realtime_decoding_csp,
)

from src.visualisation import (
    plot_realtime_predictions,
    plot_latency_over_time,
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading multimodal EEG CSV...")
    df = load_csv(CSV_PATH)

    print("Creating MNE Raw object...")
    raw = create_raw_from_dataframe(
        df=df,
        eeg_channels=EEG_CHANNELS,
        eog_channels=EOG_CHANNELS,
        sfreq=SFREQ,
    )

    print("Filtering EEG in mu/beta range...")
    raw_filtered = filter_raw(
        raw,
        low_freq=BANDPASS_LOW,
        high_freq=BANDPASS_HIGH,
    )

    print("Creating events and epochs...")
    events = create_events_from_cues(df, EVENT_ID)

    epochs = create_epochs(
        raw_filtered,
        events,
        EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
    )

    epochs = epochs.copy().pick(DECODING_CHANNELS)

    print(f"Using decoding channels: {DECODING_CHANNELS}")
    print(epochs)

    print("Extracting sliding-window EEG data...")

    X_windows, y_windows = extract_sliding_window_data(
        epochs=epochs,
        sfreq=SFREQ,
        window_length_sec=WINDOW_LENGTH_SEC,
        step_size_sec=STEP_SIZE_SEC,
    )

    print(f"Window EEG data shape: {X_windows.shape}")
    print("Class counts:", dict(zip(*np.unique(y_windows, return_counts=True))))

    print("Training CSP on sliding-window EEG data...")

    csp = CSP(
        n_components=CSP_COMPONENTS,
        reg=None,
        log=True,
        norm_trace=False,
    )

    X_csp = csp.fit_transform(X_windows, y_windows)

    print(f"CSP feature matrix shape: {X_csp.shape}")

    print("Training classifier on CSP features...")

    classifier_model, acc, report = train_evaluate_classifier(
        X_csp,
        y_windows,
        random_state=RANDOM_STATE,
    )

    print("\nOffline window-level CSP results")
    print("--------------------------------")
    print(f"CSP window-level accuracy: {acc:.3f}")
    print(report)

    print("Running real-time CSP sliding-window simulation...")

    prediction_log = simulate_realtime_decoding_csp(
        epochs=epochs,
        classifier_model=classifier_model,
        csp_model=csp,
        decoding_channels=DECODING_CHANNELS,
        sfreq=SFREQ,
        window_length_sec=WINDOW_LENGTH_SEC,
        step_size_sec=STEP_SIZE_SEC,
    )

    output_csv = os.path.join(
        RESULTS_DIR,
        "realtime_prediction_log.csv",
    )

    prediction_log.to_csv(output_csv, index=False)

    print(f"Prediction log saved to: {output_csv}")

    realtime_accuracy = (
        prediction_log["true_label"] == prediction_log["predicted_label"]
    ).mean()

    mean_latency = prediction_log["latency_ms"].mean()
    max_latency = prediction_log["latency_ms"].max()

    print("\nReal-time CSP simulation results")
    print("--------------------------------")
    print(f"Number of windows: {len(prediction_log)}")
    print(f"Raw accuracy: {realtime_accuracy:.3f}")
    print(f"Mean latency: {mean_latency:.3f} ms")
    print(f"Max latency: {max_latency:.3f} ms")

    unique_predictions, prediction_counts = np.unique(
        prediction_log["predicted_label"],
        return_counts=True,
    )

    print(
        "Prediction counts:",
        dict(zip(unique_predictions, prediction_counts)),
    )

    plot_realtime_predictions(
        prediction_log=prediction_log,
        results_dir=RESULTS_DIR,
    )

    plot_latency_over_time(
        prediction_log=prediction_log,
        results_dir=RESULTS_DIR,
    )

    print("\nReal-time CSP simulation complete.")


if __name__ == "__main__":
    main()