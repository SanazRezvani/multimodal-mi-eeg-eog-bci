CSV_PATH = "data/MI011.csv"
RESULTS_DIR = "results"

SFREQ = 1000  # EEG sampling rate from the paper

EEG_CHANNELS0 = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "M1", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2"
]

# bads = ['PO3', 'F1', 'POZ', 'OZ', 'F3', 'O2', 'P8', 'PO7', 'FC3', 'P7', 'P4']

EEG_CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'FZ', 'F2', 'F4', 'F6',
       'F8', 'FT7', 'FC5', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
       'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5',
       'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P5', 'P3', 'P1',
       'PZ', 'P2', 'P6', 'PO5', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'CB2',
       ]

EOG_CHANNELS = ["HEO"]

# MOTOR_CHANNELS = ["C3", "CZ", "C4"]
DECODING_CHANNELS = EEG_CHANNELS

EVENT_ID = {
    "Left": 1,
    "Right": 2,
}

TMIN = 0.0
TMAX = 4.0

BANDPASS_LOW = 8
BANDPASS_HIGH = 30

RANDOM_STATE = 42

# Real-time settings 
WINDOW_LENGTH_SEC = 1.0
STEP_SIZE_SEC = 0.25
REALTIME_START_TIME = 20
REALTIME_END_TIME = 40

CSP_COMPONENTS = 4

