"""
data_analysis.py
────────────────
Exploratory Data Analysis for Medical Equipment Management
Author: Fatimah Jamaan
Course: Introduction to Data Science — Cisco Networking Academy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA_PATH = "../data/equipment_data.csv"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "Operational": "#27AE60",
    "Idle":        "#E67E22",
    "Maintenance": "#2E75B6",
    "Faulty":      "#E74C3C",
}
BLUE_PALETTE = ["#1F3864", "#2E75B6", "#4472C4", "#70AD47", "#ED7D31", "#FFC000"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} records | {df.shape[1]} features")
    print(f"\n📋 Columns:\n{list(df.columns)}\n")
    print(df.describe(include="all").T.to_string())
    return df


def validate_data(df):
    print("\n── Data Validation ──────────────────────────────────────────────")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found.")
    else:
        print("⚠  Missing values:\n", missing[missing > 0])

    dupes = df.duplicated().sum()
    print(f"{'✅' if dupes == 0 else '⚠ '} Duplicate rows: {dupes}")

    # Range checks
    assert df["Daily_Usage_Hours"].between(0, 24).all(),  "Usage hours out of range!"
    assert df["Temperature_C"].between(0, 120).all(),     "Temperature out of range!"
    print("✅ All numeric range checks passed.")
    print("─" * 60)


def plot_status_distribution(df):
    counts = df["Current_Status"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Equipment Status Distribution", fontsize=14, fontweight="bold")

    # Pie
    colours = [COLORS[s] for s in counts.index]
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=colours, startangle=90, pctdistance=0.8)
    axes[0].set_title("Proportion by Status")

    # Bar
    bars = axes[1].bar(counts.index, counts.values, color=colours, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(val), ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Number of Devices")
    axes[1].set_title("Count by Status")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_status_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 01_status_distribution.png")


def plot_usage_by_device(df):
    grouped = (df.groupby("Device_Type")[["Daily_Usage_Hours", "Daily_Idle_Hours"]]
                 .mean().sort_values("Daily_Usage_Hours", ascending=True))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(grouped))
    ax.barh(y, grouped["Daily_Usage_Hours"], color="#2E75B6", label="Avg Usage (hrs)", height=0.5)
    ax.barh(y, grouped["Daily_Idle_Hours"],  color="#ED7D31", label="Avg Idle (hrs)",
            left=grouped["Daily_Usage_Hours"], height=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(grouped.index)
    ax.set_xlabel("Hours per Day")
    ax.set_title("Average Daily Usage vs Idle Hours by Device Type", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_usage_by_device.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 02_usage_by_device.png")


def plot_department_overview(df):
    dept = df.groupby("Department").agg(
        Device_Count=("Device_ID", "count"),
        Avg_Usage=("Daily_Usage_Hours", "mean"),
        Avg_Errors=("Error_Count_Monthly", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Department Overview", fontsize=14, fontweight="bold")

    for ax, (col, title, color) in zip(axes, [
        ("Device_Count", "Device Count",          "#1F3864"),
        ("Avg_Usage",    "Avg Daily Usage (hrs)",  "#2E75B6"),
        ("Avg_Errors",   "Avg Monthly Errors",     "#E74C3C"),
    ]):
        bars = ax.bar(dept["Department"], dept[col], color=color, edgecolor="white")
        for bar, val in zip(bars, dept[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_department_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 03_department_overview.png")


def plot_maintenance_risk(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Maintenance Risk Factors", fontsize=14, fontweight="bold")

    # Temperature vs Error scatter
    colors = [COLORS[s] for s in df["Current_Status"]]
    axes[0].scatter(df["Temperature_C"], df["Error_Count_Monthly"],
                    c=colors, alpha=0.6, edgecolors="white", linewidth=0.5, s=50)
    axes[0].set_xlabel("Operating Temperature (°C)")
    axes[0].set_ylabel("Monthly Error Count")
    axes[0].set_title("Temperature vs Errors")
    patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items()]
    axes[0].legend(handles=patches, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Days since maintenance histogram
    need = df[df["Needs_Maintenance"] == 1]["Days_Since_Last_Maintenance"]
    ok   = df[df["Needs_Maintenance"] == 0]["Days_Since_Last_Maintenance"]
    axes[1].hist(ok,   bins=20, color="#27AE60", alpha=0.7, label="OK",             edgecolor="white")
    axes[1].hist(need, bins=20, color="#E74C3C", alpha=0.7, label="Needs Maint.",   edgecolor="white")
    axes[1].set_xlabel("Days Since Last Maintenance")
    axes[1].set_ylabel("Number of Devices")
    axes[1].set_title("Maintenance Urgency")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_maintenance_risk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 04_maintenance_risk.png")


def summary_report(df):
    print("\n══ Summary Report ══════════════════════════════════════════════════")
    print(f"  Total devices tracked          : {len(df)}")
    print(f"  Operational                    : {(df['Current_Status']=='Operational').sum()}")
    print(f"  Needs maintenance (ML flag)    : {df['Needs_Maintenance'].sum()}")
    print(f"  Faulty devices                 : {(df['Current_Status']=='Faulty').sum()}")
    print(f"  Avg daily usage hours          : {df['Daily_Usage_Hours'].mean():.1f} hrs")
    print(f"  Avg monthly error count        : {df['Error_Count_Monthly'].mean():.1f}")
    print(f"  Most common device             : {df['Device_Type'].mode()[0]}")
    print(f"  Highest-risk department        : {df.groupby('Department')['Needs_Maintenance'].mean().idxmax()}")
    print("═" * 60)


if __name__ == "__main__":
    df = load_data()
    validate_data(df)
    plot_status_distribution(df)
    plot_usage_by_device(df)
    plot_department_overview(df)
    plot_maintenance_risk(df)
    summary_report(df)
    print("\n✅ All analysis complete. Figures saved to outputs/figures/")
