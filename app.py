from flask import Flask, render_template, request, jsonify
from model_util import get_model, load_model
import pandas as pd
import os
import random

app = Flask(__name__)

# 🎯 10 fitur paling berpengaruh
FEATURES = [
    "OverTime", "StockOptionLevel", "JobLevel", "EnvironmentSatisfaction",
    "JobInvolvement", "MaritalStatus", "JobSatisfaction", "JobRole",
    "BusinessTravel", "Age"
]

# ✅ Nilai default netral (biar prediksi stabil)
DEFAULT_VALUES = {
    "Department_Research & Development": 1,
    "EducationField_Life Sciences": 1,
    "Gender_Male": 1,
    "WorkLifeBalance": 3,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "TrainingTimesLastYear": 2,
    "StandardHours": 80,
    "Over18_Y": 1,
    "EmployeeCount": 1,
    "PercentSalaryHike": 12,
    "DistanceFromHome": 10,
    "NumCompaniesWorked": 3,
    "YearsSinceLastPromotion": 2,
    "TotalWorkingYears": 10,
    "YearsAtCompany": 7,
    "MonthlyIncome": 6000
}

# 🏠 Route utama
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def get_dashboard():
    return render_template('dashboard_view.html')

@app.route('/model')
def get_model_new():
    model = get_model()
    print("✅ Model downloaded and saved successfully." if model else "❌ Failed to download and save model.")
    return render_template('home.html')

# ⚡ Route Prediksi (utama)
@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    model = load_model()
    expected_features = model.get_booster().feature_names
    prediction, probability = None, None

    # ✅ Saat pertama kali dibuka → generate nilai acak
    if request.method == "GET":
        features = {
            "Age": random.randint(25, 45),
            "StockOptionLevel": random.randint(0, 3),
            "JobLevel": random.randint(1, 5),
            "EnvironmentSatisfaction": random.randint(1, 4),
            "JobInvolvement": random.randint(1, 4),
            "JobSatisfaction": random.randint(1, 4),
            "OverTime": random.choice(["Yes", "No"]),
            "MaritalStatus": random.choice(["Single", "Married", "Divorced"]),
            "JobRole": random.choice([
                "Human Resources", "Healthcare Representative", "Research Scientist",
                "Sales Executive", "Manager", "Laboratory Technician",
                "Research Director", "Manufacturing Director", "Sales Representative"
            ]),
            "BusinessTravel": random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        }
    else:
        # 🧩 Ambil input dari form
        form = request.form
        features = {f: form.get(f) for f in FEATURES}

        # --- Encoding kategori jadi format model ---
        encoded = {}

        # 1️⃣ OverTime
        encoded["OverTime_Yes"] = 1.0 if features["OverTime"] == "Yes" else 0.0
        encoded["OverTime_No"] = 1.0 if features["OverTime"] == "No" else 0.0

        # 2️⃣ Marital Status
        ms = features["MaritalStatus"]
        encoded["MaritalStatus_Single"] = 1.0 if ms == "Single" else 0.0
        encoded["MaritalStatus_Married"] = 1.0 if ms == "Married" else 0.0
        encoded["MaritalStatus_Divorced"] = 1.0 if ms == "Divorced" else 0.0

        # 3️⃣ Job Role
        jr = features["JobRole"]
        all_roles = [
            "Human Resources", "Healthcare Representative", "Research Scientist",
            "Sales Executive", "Manager", "Laboratory Technician",
            "Research Director", "Manufacturing Director", "Sales Representative"
        ]
        for role in all_roles:
            encoded[f"JobRole_{role}"] = 1.0 if jr == role else 0.0

        # 4️⃣ Business Travel
        bt = features["BusinessTravel"]
        encoded["BusinessTravel_Non-Travel"] = 1.0 if bt == "Non-Travel" else 0.0
        encoded["BusinessTravel_Travel_Rarely"] = 1.0 if bt == "Travel_Rarely" else 0.0
        encoded["BusinessTravel_Travel_Frequently"] = 1.0 if bt == "Travel_Frequently" else 0.0

        # 5️⃣ Numeric features
        encoded["Age"] = float(features["Age"])
        encoded["StockOptionLevel"] = float(features["StockOptionLevel"])
        encoded["JobLevel"] = float(features["JobLevel"])
        encoded["EnvironmentSatisfaction"] = float(features["EnvironmentSatisfaction"])
        encoded["JobInvolvement"] = float(features["JobInvolvement"])
        encoded["JobSatisfaction"] = float(features["JobSatisfaction"])

        # Gabungkan dengan fitur default
        df = pd.DataFrame([encoded])
        for col in expected_features:
            if col not in df.columns:
                df[col] = DEFAULT_VALUES.get(col, 0)

        df = df[expected_features]

        # --- Prediksi ---
        try:
            pred_proba = model.predict_proba(df)[0]
            prob_yes = float(pred_proba[1]) * 100
            prob_no = float(pred_proba[0]) * 100

            pred = 1 if prob_yes > 50 else 0  # threshold default 50%
            prediction = "Yes" if pred == 1 else "No"
            probability = f"{prob_yes:.2f}%" if pred == 1 else f"{prob_no:.2f}%"
        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = None

    return render_template("predict_view.html", features=features, prediction=prediction, probability=probability)

# 🔹 API Endpoint (optional)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    model = load_model()
    expected_features = model.get_booster().feature_names

    # Encoding sama seperti form
    encoded = {}

    encoded["OverTime_Yes"] = 1.0 if data.get("OverTime") == "Yes" else 0.0
    encoded["OverTime_No"] = 1.0 if data.get("OverTime") == "No" else 0.0

    ms = data.get("MaritalStatus")
    encoded["MaritalStatus_Single"] = 1.0 if ms == "Single" else 0.0
    encoded["MaritalStatus_Married"] = 1.0 if ms == "Married" else 0.0
    encoded["MaritalStatus_Divorced"] = 1.0 if ms == "Divorced" else 0.0

    jr = data.get("JobRole")
    for role in [
        "Human Resources", "Healthcare Representative", "Research Scientist",
        "Sales Executive", "Manager", "Laboratory Technician",
        "Research Director", "Manufacturing Director", "Sales Representative"
    ]:
        encoded[f"JobRole_{role}"] = 1.0 if jr == role else 0.0

    bt = data.get("BusinessTravel")
    encoded["BusinessTravel_Non-Travel"] = 1.0 if bt == "Non-Travel" else 0.0
    encoded["BusinessTravel_Travel_Rarely"] = 1.0 if bt == "Travel_Rarely" else 0.0
    encoded["BusinessTravel_Travel_Frequently"] = 1.0 if bt == "Travel_Frequently" else 0.0

    # Numeric
    encoded["Age"] = float(data.get("Age", 35))
    encoded["StockOptionLevel"] = float(data.get("StockOptionLevel", 1))
    encoded["JobLevel"] = float(data.get("JobLevel", 2))
    encoded["EnvironmentSatisfaction"] = float(data.get("EnvironmentSatisfaction", 3))
    encoded["JobInvolvement"] = float(data.get("JobInvolvement", 3))
    encoded["JobSatisfaction"] = float(data.get("JobSatisfaction", 3))

    df = pd.DataFrame([encoded])
    for col in expected_features:
        if col not in df.columns:
            df[col] = DEFAULT_VALUES.get(col, 0)
    df = df[expected_features]

    pred_proba = model.predict_proba(df)[0]
    prob_yes = float(pred_proba[1]) * 100
    prob_no = float(pred_proba[0]) * 100
    pred = 1 if prob_yes > 50 else 0

    prediction = "Yes" if pred == 1 else "No"
    probability = f"{prob_yes:.2f}%" if pred == 1 else f"{prob_no:.2f}%"

    return jsonify({
        "prediction": prediction,
        "confidence": probability,
        "features": data
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
