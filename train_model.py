# train_model.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)

# ============================================
# Load Dataset
# ============================================

print("=" * 60)
print("         AI CYBER THREAT DETECTION")
print("=" * 60)

dataset = pd.read_csv("cyber_dataset.csv")

print("\n[+] Dataset Loaded Successfully")
print(f"Total Records : {len(dataset)}")
print(f"Total Columns : {len(dataset.columns)}")

# ============================================
# Separate Features & Target
# ============================================

X = dataset.drop("attack_type", axis=1)
y = dataset["attack_type"]

# ============================================
# One Hot Encoding
# ============================================

X = pd.get_dummies(X, columns=["protocol"])
feature_columns = X.columns.tolist()

# ============================================
# Encode Target Labels
# ============================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================
# Algorithms Definitions
# ============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
best_model = None
best_algorithm = ""
best_accuracy = 0

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# ============================================
# Train & Evaluate Models
# ============================================

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    loss = log_loss(y_test, y_prob)
    
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": loss
    }
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_algorithm = name

# ============================================
# Dynamic Accuracy Comparison Graph
# ============================================

algorithms = list(results.keys())
accuracies = [results[algo]["accuracy"] * 100 for algo in algorithms]

plt.figure(figsize=(8, 5))
colors = ["green" if algo == best_algorithm else "steelblue" for algo in algorithms]

bars = plt.bar(algorithms, accuracies, color=colors, width=0.55)
plt.title("Algorithm Accuracy Comparison", fontsize=14)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1.5,
        f"{height:.2f}%",
        ha="center",
        fontsize=10,
        weight="bold"
    )

plt.tight_layout()
plt.savefig("accuracy_graph.png")
plt.close()

# ============================================
# Save Best Model & Structural Metadata
# ============================================

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# EXPORT THE METRICS PACK FOR FLASK READABILITY
best_metrics = {
    "algorithm": best_algorithm,
    "accuracy": f"{results[best_algorithm]['accuracy']*100:.2f}%",
    "precision": f"{results[best_algorithm]['precision']:.4f}",
    "recall": f"{results[best_algorithm]['recall']:.4f}",
    "f1": f"{results[best_algorithm]['f1']:.4f}",
    "loss": f"{results[best_algorithm]['loss']:.4f}"
}

with open("metrics.pkl", "wb") as f:
    pickle.dump(best_metrics, f)

print("\n[+] Training Complete. All Artifacts and Metrics Successfully Serialized.")