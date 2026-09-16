import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Paths relative to this file, so the script works on any computer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "uci-ml-phishing-dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_model.pkl")

# Only features that can be computed from a URL alone.
# The other UCI features need the page HTML, WHOIS data or traffic stats,
# which the API does not have when it receives a URL.
# These names must match the keys returned by extract_features() in backend/app.py.
FEATURES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "port",
    "HTTPS_token",
]

# Load dataset (labels: -1 = phishing, 1 = legitimate)
data = pd.read_csv(DATASET_PATH)
X = data[FEATURES]
y = data["Result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Model Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(classification_report(y_test, predictions, target_names=["Phishing", "Legitimate"]))

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully to", MODEL_PATH)
