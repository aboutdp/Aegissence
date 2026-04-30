import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_lstm(uploaded_file):

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(uploaded_file)

    # ---------------- SELECT NUMERIC ----------------
    df_numeric = df.select_dtypes(include=[np.number])

    # ---------------- LIMIT DATA (rows) ----------------
    df_numeric = df_numeric.head(9790)

    # ---------------- NORMALIZATION ----------------
    df_norm = (df_numeric - df_numeric.mean()) / (df_numeric.std() + 1e-8)

    # ---------------- RESHAPE ----------------
    X = df_norm.values.reshape((df_norm.shape[0], 1, df_norm.shape[1]))

    # ---------------- MODEL ----------------
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(1, X.shape[2])))
    model.add(Dense(X.shape[2]))
    model.compile(optimizer='adam', loss='mse')

    # ---------------- TRAIN ----------------
    model.fit(X, X, epochs=3, batch_size=32, verbose=0)

    # ---------------- PREDICT (BATCHED) ----------------
    preds = model.predict(X, batch_size=32, verbose=0)

    # ---------------- ERROR CALCULATION ----------------
    error = np.mean((X - preds) ** 2, axis=(1, 2))

    # ---------------- THRESHOLD ----------------
    threshold = np.percentile(error, 95)

    # ---------------- OUTPUT ----------------
    df_numeric["anomaly"] = (error > threshold).astype(int)
    df_numeric["anomaly_score"] = error

    return df_numeric