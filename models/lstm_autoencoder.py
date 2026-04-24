import numpy as np
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from data.preprocess import preprocess


# Convert data into sequences
def create_sequences(data, seq_length=20):  
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)


def run_lstm(file_path=None):
    data, df = preprocess(file_path)

    # Create sequences
    X = create_sequences(data)

    timesteps = X.shape[1]
    features = X.shape[2]

    # Model architecture
    inputs = Input(shape=(timesteps, features))
    encoded = LSTM(64, activation='relu')(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(features, activation='relu', return_sequences=True)(decoded)

    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')

    # ---------------- SAVE / LOAD MODEL ----------------
    model_path = "models/lstm_model.h5"

    if os.path.exists(model_path):
        print("✅ Loading saved model...")
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse')
    else:
        print("🚀 Training new model...")
        model.fit(X, X, epochs=2, batch_size=32, verbose=1)
        model.save(model_path)
        print("💾 Model saved!")

    # ---------------- PREDICTION ----------------
    recon = model.predict(X)

    # Error calculation
    mse = np.mean(np.power(X - recon, 2), axis=(1, 2))

    # Threshold
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold

    # Align with dataframe
    df = df.iloc[len(df) - len(anomalies):].copy()
    df["anomaly"] = anomalies.astype(int)

    return df


if __name__ == "__main__":
    df = run_lstm()
    print(df.head())