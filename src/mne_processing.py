import numpy as np
import pandas as pd
import mne


def create_raw_from_dataframe(df, eeg_channels, eog_channels, sfreq):
    all_channels = eeg_channels + eog_channels

    data = df[all_channels].to_numpy().T

    ch_types = ["eeg"] * len(eeg_channels) + ["eog"] * len(eog_channels)

    info = mne.create_info(
        ch_names=all_channels,
        sfreq=sfreq,
        ch_types=ch_types,
    )

    raw = mne.io.RawArray(data, info)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    return raw


def filter_raw(raw, low_freq, high_freq):
    raw_filtered = raw.copy()
    raw_filtered.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        fir_design="firwin",
        verbose=False,
    )
    return raw_filtered


import numpy as np
import pandas as pd


def create_events_from_cues(df, event_id):
    """
    Create one event per motor imagery trial.

    The CSV contains many rows where Cues may be 'Left' or 'Right'.
    We only want the first sample of each trial, not every row.
    """

    events = []

    cues = df["Cues"].astype(str).str.strip()
    trig = pd.to_numeric(df["Trig"], errors="coerce")

    df_temp = df.copy()
    df_temp["Cues_clean"] = cues
    df_temp["Trig_numeric"] = trig

    # Keep only rows with valid MI cues
    mi_rows = df_temp[df_temp["Cues_clean"].isin(event_id.keys())].copy()

    # Group by trial number and take the first cue row per trial
    for trial_id, trial_df in mi_rows.groupby("Trig_numeric"):
        if pd.isna(trial_id):
            continue

        first_row_index = trial_df.index[0]
        cue = trial_df.loc[first_row_index, "Cues_clean"]

        events.append([
            int(first_row_index),   # sample index
            0,
            event_id[cue]
        ])

    events = np.array(events, dtype=int)

    return events


def create_epochs(raw, events, event_id, tmin, tmax):
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs