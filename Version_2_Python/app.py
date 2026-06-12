"""
app.py — Streamlit Dashboard
─────────────────────────────
Smart Medical Equipment Management System
Author: Fatimah Jamaan
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Medical Equipment Manager",
    page_icon="🏥",
    layout="wide",
)

COLORS = {
    "Operational": "#27AE60",
    "Idle":        "#E67E22",
    "Maintenance": "#2E75B6",
    "Faulty":      "#E74C3C",
}
FEATURES = [
    "Age_Years", "Daily_Usage_Hours", "Daily_Idle_Hours",
    "Total_Runtime_Hours", "Temperature_C",
    "Error_Count_Monthly", "Days_Since_Last_Maintenance",
]

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("../data/equipment_data.csv")

@st.cache_resource
def load_model():
    model_path  = "outputs/models/best_model.pkl"
    scaler_path = "outputs/models/scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

df    = load_data()
model, scaler = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=80)
st.sidebar.title("🏥 Medical Equipment\nManager")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🔍 Usage Analysis",
    "🔧 Maintenance Tracker",
    "🤖 ML Prediction",
])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Equipment Overview Dashboard")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔢 Total Devices",      len(df))
    c2.metric("✅ Operational",        (df["Current_Status"]=="Operational").sum())
    c3.metric("⚠️ Needs Maintenance",  df["Needs_Maintenance"].sum())
    c4.metric("❌ Faulty",             (df["Current_Status"]=="Faulty").sum())

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Status pie
    with col_l:
        st.subheader("Equipment Status Distribution")
        counts  = df["Current_Status"].value_counts()
        colours = [COLORS[s] for s in counts.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=colours, startangle=90)
        ax.set_title("Current Status")
        st.pyplot(fig)
        plt.close()

    # Devices per department
    with col_r:
        st.subheader("Devices per Department")
        dept_counts = df["Department"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(dept_counts.index, dept_counts.values, color="#2E75B6")
        ax.set_xlabel("Number of Devices")
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, dept_counts.values):
            ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                    str(val), va="center")
        st.pyplot(fig)
        plt.close()

    # Data table with filter
    st.markdown("---")
    st.subheader("📋 Raw Data Explorer")
    dept_filter   = st.multiselect("Filter by Department", df["Department"].unique(), default=list(df["Department"].unique()))
    status_filter = st.multiselect("Filter by Status",     df["Current_Status"].unique(), default=list(df["Current_Status"].unique()))
    filtered = df[df["Department"].isin(dept_filter) & df["Current_Status"].isin(status_filter)]
    st.dataframe(filtered, use_container_width=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} records")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — USAGE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Usage Analysis":
    st.title("🔍 Equipment Usage Analysis")
    st.markdown("---")

    grouped = (df.groupby("Device_Type")[["Daily_Usage_Hours","Daily_Idle_Hours"]]
                 .mean().sort_values("Daily_Usage_Hours", ascending=True))

    st.subheader("Average Daily Usage vs Idle Hours")
    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(grouped))
    ax.barh(list(y), grouped["Daily_Usage_Hours"], color="#1F3864", label="Usage", height=0.5)
    ax.barh(list(y), grouped["Daily_Idle_Hours"],  color="#ED7D31", label="Idle",
            left=grouped["Daily_Usage_Hours"], height=0.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(grouped.index)
    ax.set_xlabel("Hours / Day"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Department-Level Statistics")
    dept_stats = df.groupby("Department").agg(
        Devices      =("Device_ID","count"),
        Avg_Usage    =("Daily_Usage_Hours","mean"),
        Avg_Idle     =("Daily_Idle_Hours","mean"),
        Avg_Errors   =("Error_Count_Monthly","mean"),
        Pct_Faulty   =("Current_Status", lambda x: (x=="Faulty").mean()*100),
    ).round(2)
    st.dataframe(dept_stats.style.background_gradient(cmap="Blues"), use_container_width=True)

    # Manufacturer comparison
    st.markdown("---")
    st.subheader("Manufacturer Reliability")
    mfr = df.groupby("Manufacturer").agg(
        Total=("Device_ID","count"),
        Avg_Errors=("Error_Count_Monthly","mean"),
        Pct_Needs_Maint=("Needs_Maintenance","mean"),
    ).round(3)
    mfr["Pct_Needs_Maint"] = (mfr["Pct_Needs_Maint"]*100).round(1)
    st.dataframe(mfr.style.highlight_max(color="#FADBD8").highlight_min(color="#D5F5E3"),
                 use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MAINTENANCE TRACKER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Maintenance Tracker":
    st.title("🔧 Predictive Maintenance Tracker")
    st.markdown("---")

    risk_df = df.copy()
    risk_df["Risk Level"] = risk_df.apply(
        lambda r: "🔴 High"   if r["Needs_Maintenance"]==1 and r["Days_Since_Last_Maintenance"]>180
             else "🟠 Medium" if r["Needs_Maintenance"]==1
             else "🟢 Low", axis=1
    )

    # Filter
    risk_filter = st.multiselect("Filter by Risk Level",
                                  ["🔴 High","🟠 Medium","🟢 Low"],
                                  default=["🔴 High","🟠 Medium"])
    show = risk_df[risk_df["Risk Level"].isin(risk_filter)].sort_values(
        "Days_Since_Last_Maintenance", ascending=False
    )

    cols = ["Device_ID","Device_Type","Department","Age_Years",
            "Days_Since_Last_Maintenance","Error_Count_Monthly",
            "Temperature_C","Current_Status","Risk Level"]
    st.dataframe(show[cols], use_container_width=True)
    st.caption(f"{len(show)} devices shown")

    # Scatter: days since maint vs errors
    st.markdown("---")
    st.subheader("Risk Factor Map")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_plot = ["#E74C3C" if v==1 else "#27AE60" for v in df["Needs_Maintenance"]]
    ax.scatter(df["Days_Since_Last_Maintenance"], df["Error_Count_Monthly"],
               c=colors_plot, alpha=0.6, s=60, edgecolors="white")
    ax.set_xlabel("Days Since Last Maintenance")
    ax.set_ylabel("Monthly Error Count")
    ax.set_title("Maintenance Urgency Map")
    patches = [mpatches.Patch(color="#E74C3C", label="Needs Maintenance"),
               mpatches.Patch(color="#27AE60", label="OK")]
    ax.legend(handles=patches)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Prediction":
    st.title("🤖 ML Maintenance Prediction")
    st.markdown("Enter device parameters to predict if maintenance is needed.")
    st.markdown("---")

    if model is None:
        st.warning("⚠️ Model not found. Please run `ml_prediction.py` first to train and save the model.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            age          = st.slider("Device Age (years)",          0.5, 15.0, 5.0, 0.5)
            usage_hours  = st.slider("Daily Usage Hours",           1, 20, 8)
            idle_hours   = st.slider("Daily Idle Hours",            0, 20, 6)
            runtime      = st.slider("Total Runtime Hours",         100, 9000, 3000, 100)
        with col2:
            temperature  = st.slider("Operating Temperature (°C)",  20.0, 100.0, 45.0, 0.5)
            errors       = st.slider("Monthly Error Count",         0, 15, 2)
            days_maint   = st.slider("Days Since Last Maintenance",  1, 365, 90)

        input_data = np.array([[age, usage_hours, idle_hours, runtime,
                                temperature, errors, days_maint]])
        input_scaled = scaler.transform(input_data)

        if st.button("🔍 Predict Maintenance Need", type="primary"):
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠️ **MAINTENANCE REQUIRED** — Risk probability: {probability:.1%}")
                st.markdown("**Recommendation:** Schedule maintenance within 48 hours.")
            else:
                st.success(f"✅ **DEVICE IS OK** — Risk probability: {probability:.1%}")
                st.markdown("**Recommendation:** Continue normal monitoring.")

            # Gauge-style bar
            st.markdown("#### Risk Level")
            st.progress(float(probability))
            st.caption(f"Maintenance probability: {probability:.1%}")
