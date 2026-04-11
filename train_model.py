import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

def log(message):
    print(message, flush=True)

# Load dataset
log("Loading dataset (this may take a moment)...")
df = pd.read_csv("creditcard.csv")

# Features & target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handling Class Imbalance using SMOTE
log("Balancing dataset using SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scaling
log("Scaling features...")
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Define models to compare (Optimized for speed)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42, max_iter=100),
    "Random Forest": RandomForestClassifier(n_estimators=30, random_state=42, max_depth=8, n_jobs=-1)
}

best_model = None
best_f1 = 0
best_model_name = ""

log("\n--- Starting Model Comparison ---")
for name, model in models.items():
    log(f"Training {name}...")
    model.fit(X_train_res, y_train_res)
    
    # Predicting on test set
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    log(f"{name}: F1-Score = {f1:.4f}, Accuracy = {acc:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

log(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")

# Save the best model and scaler
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save model metadata for the app
with open("model_metadata.pkl", "wb") as f:
    pickle.dump({"name": best_model_name, "f1_score": best_f1}, f)

log("\nModel and metadata saved successfully!")