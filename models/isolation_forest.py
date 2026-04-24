from sklearn.ensemble import IsolationForest
from data.preprocess import preprocess

def run_model():
    # Load processed data
    data, df = preprocess()

    # Create model
    model = IsolationForest(contamination=0.03, random_state=42)

    # Train + predict
    preds = model.fit_predict(data)

    # Convert predictions
    df["anomaly"] = [1 if p == -1 else 0 for p in preds]

    return df


if __name__ == "__main__":
    df = run_model()
    print(df.head())