from pathlib import Path

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model_path = Path(__file__).with_name("heart_model.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("result.html", prediction="Model file not found. Train and save heart_disease_model.pkl to use prediction.")

    values = []
    for i in range(13):
        values.append(float(request.form.get(f"feature_{i}", 0)))

    features = np.array(values).reshape(1, -1)
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "The patient is likely to have heart disease."
    else:
        result = "The patient is unlikely to have heart disease."

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)