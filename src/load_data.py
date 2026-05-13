import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Standardise missing values
    df = df.replace("NA", pd.NA)

    # Convert numeric columns where possible
    for col in df.columns:
        if col not in ["Cues", "PhanTime", "RecordingTimestamp", "LocalTimeStamp"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_motor_imagery_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Cues"].isin(["Left", "Right"])].copy()