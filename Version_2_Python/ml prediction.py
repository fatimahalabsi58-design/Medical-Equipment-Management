"""
ml_prediction.py
────────────────
Machine Learning — Predictive Maintenance for Medical Equipment
Author: Fatimah Jamaan
Course: Introduction to Data Science — Cisco Networking Academy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, joblib

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (accuracy_score, classification_report,
                                        confusion_matrix, roc_auc_score, roc_curve)

DATA_PATH  = "../data/equipment_data.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR  = "outputs/models"
FIG_DIR    = "outputs/figures"
for d in [OUTPUT_DIR, MODEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES = [
    "Age_Years", "Daily_Usage_Hours", "Daily_Idle_Hours",
    "Total_Runtime_Hours", "Temperature_C",
    "Error_Count_Monthly", "Days_Since_Last_Maintenance",
]
TARGET = "Needs_Maintenance"


# ── 1. Load & Prepare ────────────────────────────────────────────────────────

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, df


# ── 2. Train Models ──────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    trained = {}
    print("\n── Cross-Validation Scores (5-fold) ──────────────────────────────")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  {name:<25} CV Acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return trained


# ── 3. Evaluate ──────────────────────────────────────────────────────────────

def evaluate_models(trained, X_test, y_test):
    results = {}
    print("\n── Test Set Results ──────────────────────────────────────────────")
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = {"model": model, "acc": acc, "auc": auc,
                         "y_pred": y_pred,
                         "y_prob": model.predict_proba(X_test)[:, 1]}
        print(f"\n  {name}")
        print(f"  Accuracy: {acc:.3f}  |  AUC-ROC: {auc:.3f}")
        print(classification_report(y_test, y_pred,
                                     target_names=["OK", "Needs Maintenance"]))
    return results


# ── 4. Plots ─────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["OK","Maint."]); ax.set_yticklabels(["OK","Maint."])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 05_confusion_matrices.png")


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1F3864", "#27AE60", "#E74C3C"]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {res['auc']:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 06_roc_curves.png")


def plot_feature_importance(best_model):
    importances = best_model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([FEATURES[i] for i in idx], importances[idx],
                   color="#2E75B6", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Random Forest", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 07_feature_importance.png")


def plot_model_comparison(results):
    names  = list(results.keys())
    accs   = [results[n]["acc"] for n in names]
    aucs   = [results[n]["auc"] for n in names]
    x      = np.arange(len(names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#1F3864")
    b2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC",  color="#ED7D31")

    for bar, val in zip(list(b1)+list(b2), accs+aucs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score"); ax.set_title("Model Comparison", fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved: 08_model_comparison.png")


# ── 5. Save Best Model ───────────────────────────────────────────────────────

def save_best_model(results, scaler):
    best_name  = max(results, key=lambda n: results[n]["auc"])
    best_model = results[best_name]["model"]
    print(f"\n🏆 Best model: {best_name} (AUC = {results[best_name]['auc']:.3f})")
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/scaler.pkl")
    print(f"💾 Saved to {MODEL_DIR}/")
    return best_model


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df = load_and_prepare()
    trained = train_models(X_train, y_train)
    results = evaluate_models(trained, X_test, y_test)

    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_model_comparison(results)

    best = save_best_model(results, scaler)
    plot_feature_importance(best)

    print("\n✅ ML pipeline complete!")
