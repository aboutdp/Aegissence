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

        if model_choice == "Isolation Forest":
            from models.isolation_forest import run_model
            st.session_state.df = run_model()

        else:
            from models.lstm_autoencoder import run_lstm
            if uploaded_file is not None:
                st.session_state.df = run_lstm(uploaded_file)
            else:
                st.session_state.df = run_lstm()

    end_time = time.time()
    st.success("✅ Model Run Completed!")
    st.info(f"⏱️ Processing Time: {round(end_time - start_time, 2)} seconds")

# ---------------- USE STORED DATA ----------------
df = st.session_state.df

# ---------------- WAIT MESSAGE ----------------
if df is None:
    st.warning("⚠️ Please click '🚀 Run Model' to start analysis")

# ---------------- MAIN OUTPUT ----------------
if df is not None:

    # Dataset indicator
    if uploaded_file is not None:
        dataset_name = uploaded_file.name
    else:
        dataset_name = "CMAPSS (Default)"

    st.info(f"📂 Dataset: {dataset_name}")

    # -------- SENSOR DETECTION --------
    sensor_columns = [col for col in df.columns if "sensor" in col]

    if len(sensor_columns) == 0:
        sensor_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if "anomaly" in sensor_columns:
        sensor_columns.remove("anomaly")
    if "anomaly_score" in sensor_columns:
        sensor_columns.remove("anomaly_score")

    # ---------------- KPI ----------------
    total_anomalies = df["anomaly"].sum()
    total_points = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Data Points", total_points)
    col2.metric("⚠️ Anomalies", total_anomalies)
    col3.metric("✅ Health", "Good" if total_anomalies < 1000 else "Warning")

    # ---------------- ALERT ----------------
    if total_anomalies > 1000:
        st.error("🚨 High Risk Detected")
    else:
        st.success("✅ System Stable")

    # ---------------- SYSTEM INSIGHT ----------------
    st.subheader("📌 System Insight")

    if total_anomalies > 1000:
        st.write("⚠️ High anomaly rate detected. Possible system instability.")
    else:
        st.write("✅ System operating within normal conditions.")

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
        label="Anomaly",
        s=20
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
st.subheader("About AegisSense")

st.write("""
AegisSense is an AI-based industrial monitoring system designed to detect anomalies 
in multi-sensor time-series data using Machine Learning and Deep Learning models.
""")

# ---------------- AUTO REFRESH ----------------
st.sidebar.subheader("🔄 Live Mode")

refresh = st.sidebar.checkbox("Enable Auto Refresh")

if refresh:
    time.sleep(3)
    st.rerun()