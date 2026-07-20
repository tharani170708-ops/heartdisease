import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from collections import Counter

# -----------------------------
# Load dataset
# -----------------------------
# Replace 'disease_data.csv' with your dataset
data = pd.read_csv("disease_data.csv")

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# -----------------------------
# Single man's data
# Example:
# Age, BMI, BloodPressure, Cholesterol, BloodSugar
# Change these values to match your dataset columns
# -----------------------------
new_man = np.array([[45, 27.5, 135, 220, 110]])

# -----------------------------
# Bootstrap Prediction
# -----------------------------
n_bootstrap = 500
predictions = []

for i in range(n_bootstrap):

    # Bootstrap sample
    X_boot, y_boot = resample(
        X,
        y,
        replace=True,
        random_state=i
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_boot, y_boot)

    # Predict
    pred = model.predict(new_man)[0]
    predictions.append(pred)

# -----------------------------
# Final Prediction
# -----------------------------
vote = Counter(predictions)
final_prediction = vote.most_common(1)[0][0]
confidence = vote[final_prediction] / n_bootstrap * 100

print("Bootstrap Predictions:", vote)

if final_prediction == 1:
    print("Prediction: Disease Detected")
else:
    print("Prediction: No Disease")

print(f"Confidence: {confidence:.2f}%")