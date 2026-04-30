import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---------------- SESSION STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AegisSense", layout="wide")

st.title("🏭 AegisSense Industrial Monitoring System")
st.markdown("### AI-based Multi-Sensor Anomaly Detection")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "🧠 Select Model",
    ["Isolation Forest", "LSTM"]
)

run_button = st.sidebar.button("🚀 Run Model")

# ---------------- DATA PREVIEW ----------------
if uploaded_file is not None:
    try:
        df_preview = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.write("Shape:", df_preview.shape)
        st.dataframe(df_preview.head(10))

        numeric_cols = df_preview.select_dtypes(include=["number"]).columns

        if len(numeric_cols) < 2:
            st.error("❌ Dataset must contain at least 2 numeric columns")
            st.stop()
        else:
            st.success("✅ Valid dataset detected")

        missing = df_preview.isnull().sum().sum()
        if missing > 0:
            st.warning(f"⚠️ Missing values: {missing}")
        else:
            st.success("✅ No missing values")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

# ---------------- RUN MODEL ----------------
if run_button:
    start_time = time.time()

    with st.spinner("⏳ Running Model... Please wait"):

        if uploaded_file is None:
            st.error("❌ Please upload dataset")
            st.stop()

        uploaded_file.seek(0)
        df_input = pd.read_csv(uploaded_file)

        if model_choice == "Isolation Forest":
            from sklearn.ensemble import IsolationForest

            df_numeric = df_input.select_dtypes(include=["number"])

            model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )

            model.fit(df_numeric)
            preds = model.predict(df_numeric)

            df_input["anomaly"] = (preds == -1).astype(int)

            st.session_state.df = df_input

        else:
            uploaded_file.seek(0)
            from models.lstm_autoencoder import run_lstm
            st.session_state.df = run_lstm(uploaded_file)

    end_time = time.time()
    st.success("✅ Model Run Completed!")
    st.info(f"⏱️ Processing Time: {round(end_time - start_time, 2)} seconds")

# ---------------- USE STORED DATA ----------------
df = st.session_state.df

# ---------------- WAIT MESSAGE ----------------
if df is None:
    st.warning("Please click '🚀 Run Model' to start analysis")

# ---------------- HELPER ----------------
def classify_severity(score):
    if score > 0.5:
        return "High"
    elif score > 0.2:
        return "Medium"
    else:
        return "Low"

# ---------------- MAIN OUTPUT ----------------
if df is not None:

    # Add severity safely
    if "anomaly_score" in df.columns:
        df["severity"] = df["anomaly_score"].apply(classify_severity)

    # Dataset indicator
    dataset_name = uploaded_file.name if uploaded_file else "Default"
    st.info(f"📂 Dataset: {dataset_name}")

    # -------- SENSOR DETECTION --------
    sensor_columns = [col for col in df.columns if "sensor" in col]

    if len(sensor_columns) == 0:
        sensor_columns = df.select_dtypes(include=["number"]).columns.tolist()

    for col in ["anomaly", "anomaly_score"]:
        if col in sensor_columns:
            sensor_columns.remove(col)

    # ---------------- KPI ----------------
    total_anomalies = int(df["anomaly"].sum())
    total_points = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Data Points", total_points)
    col2.metric("⚠️ Anomalies", total_anomalies)
    col3.metric("📉 Rate", f"{round(df['anomaly'].mean()*100,2)}%")

    # ---------------- SYSTEM RISK ----------------
    anomaly_rate = df["anomaly"].mean()

    if anomaly_rate > 0.1:
        st.error("🚨 System Risk: HIGH")
    elif anomaly_rate > 0.05:
        st.warning("⚠️ System Risk: MEDIUM")
    else:
        st.success("✅ System Stable")

    # ---------------- CONTROLS ----------------
    st.sidebar.header("⚙️ Controls")

    selected_sensors = st.sidebar.multiselect(
        "Select Sensors",
        sensor_columns,
        default=sensor_columns[:3]
    )

    focus_sensor = st.sidebar.selectbox(
        "Focus Sensor",
        sensor_columns
    )

    # ---------------- ZOOM ----------------
    st.subheader("🔍 Data Zoom")

    start = st.slider("Start Index", 0, len(df)-100, 0)
    end = start + 100

    df_window = df.iloc[start:end]

    # ---------------- MULTI SENSOR GRAPH ----------------
    if selected_sensors:
        st.subheader("📊 Multi-Sensor Trends")
        st.line_chart(df_window[selected_sensors])

    # ---------------- SINGLE SENSOR GRAPH ----------------
    st.subheader(f"📈 {focus_sensor} with Anomalies")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_window[focus_sensor], label=focus_sensor)

    anomalies = df_window[df_window["anomaly"] == 1]

    ax.scatter(
        anomalies.index,
        anomalies[focus_sensor],
        color='red',
        s=40,
        alpha=0.7,
        label="Anomaly"
    )

    ax.legend()
    st.pyplot(fig)

    # ---------------- ANOMALY SCORE ----------------
    if "anomaly_score" in df.columns:
        st.subheader("📉 Anomaly Score")
        st.line_chart(df_window["anomaly_score"])

    # ---------------- TABLE ----------------
    st.subheader("⚠️ Detected Anomalies")
    st.dataframe(df[df["anomaly"] == 1].head(100))

    # ---------------- EXPLAINABILITY ----------------
    st.subheader("🧠 Key Influencing Sensors")

    importance = df[sensor_columns].std().sort_values(ascending=False).head(5)
    st.bar_chart(importance)

    # ---------------- DOWNLOAD ----------------
    st.markdown("""
    <style>
    div.stDownloadButton > button {
    background-color: #ff8c00;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
    font-weight: bold;
    }
    div.stDownloadButton > button:hover {
    background-color: #e67e00;
    color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    st.download_button(
        label="Download Results",
        data=df.to_csv(index=False),
        file_name="anomaly_results.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<h3 style='text-align: center; font-weight: 800;'>AegisSense | Predictive Maintenance System</h3>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>Henal | Daksh | Ravi | Hetv | Paxaal</p>",
    unsafe_allow_html=True
)

# ---------------- AUTO REFRESH ----------------
st.sidebar.subheader("🔄 Live Mode")
refresh = st.sidebar.checkbox("Enable Auto Refresh")

if refresh:
    time.sleep(3)
    st.rerun()