import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define absolute paths to ensure scripts execute properly from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "visualizations")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")

# ---------------------------------------------------------
# Step 1: Load and Inspect Dataset
# ---------------------------------------------------------
# Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset loaded successfully from: {CSV_PATH}")
except FileNotFoundError:
    print(
        f"⚠️ File not found at '{CSV_PATH}'.\n"
        "Generating synthetic sample data for demonstration..."
    )
    np.random.seed(42)
    n_samples = 10000
    df = pd.DataFrame(
        np.random.randn(n_samples, 28),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    df["Time"] = np.random.randint(0, 172800, n_samples)
    df["Amount"] = np.random.exponential(scale=100, size=n_samples)
    df["Class"] = np.random.choice([0, 1], size=n_samples, p=[0.998, 0.002])

print(f"Dataset Shape: {df.shape}")
print("\nClass Distribution:")
print(df["Class"].value_counts())

# ---------------------------------------------------------
# Step 2: Data Cleaning & Preprocessing
# ---------------------------------------------------------
# Remove any missing values
df.dropna(inplace=True)

# Standardize Time and Amount columns
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_time"] = scaler.fit_transform(df[["Time"]])

# Drop original unscaled columns
df.drop(["Amount", "Time"], axis=1, inplace=True)

# Define Features (X) and Target Label (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# Stratified train-test split due to target imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# ---------------------------------------------------------
# Step 3: Model Training
# ---------------------------------------------------------
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 4: Model Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0

print("\n--- Evaluation Metrics ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------------
# Step 5: Visualizations
# ---------------------------------------------------------

# Plot 1: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate (0)", "Fraudulent (1)"],
    yticklabels=["Legitimate (0)", "Fraudulent (1)"],
)
plt.title("Confusion Matrix - Credit Card Fraud Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# Plot 2: ROC Curve
if len(np.unique(y_test)) > 1:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "roc_curve.png"), dpi=300)
    plt.close()

print(f"Pipeline executed successfully. Plots saved to '{VISUALS_DIR}'.")