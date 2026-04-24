import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(file_path=None):
    # If user uploads file
    if file_path:
        df = pd.read_csv(file_path)
    else:
        # Default dataset
        df = pd.read_csv("data/cmapss_combined.csv")

    # Auto-detect numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Remove obvious non-sensor columns if present
    drop_cols = ["unit", "cycle"]
    features = df[numeric_cols].drop(columns=[col for col in drop_cols if col in numeric_cols], errors="ignore")

    # Normalize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    return scaled_data, df