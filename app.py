from flask import Flask, render_template, request, jsonify
from model_util import get_model, load_model
import pandas as pd
import os
import random

app = Flask(__name__)

# 🎯 Fitur input manual oleh user (10 fitur utama sebelum encoding)
FEATURES = [
    "OverTime", "StockOptionLevel", "JobLevel", "EnvironmentSatisfaction",
    "JobInvolvement", "MaritalStatus", "JobSatisfaction", "JobRole",
    "BusinessTravel", "EducationField"
]

# 🧠 Semua fitur encoded yang dipakai model
ENCODED_FEATURES = [
    "OverTime_Yes",
    "StockOptionLevel",
    "JobLevel",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "MaritalStatus_Single", "MaritalStatus_Married", "MaritalStatus_Divorced",
    "JobSatisfaction",
    "JobRole_Human Resources", "JobRole_Healthcare Representative", "JobRole_Research Scientist",
    "JobRole_Sales Executive", "JobRole_Manager", "JobRole_Laboratory Technician",
    "JobRole_Research Director", "JobRole_Manufacturing Director", "JobRole_Sales Representative",
    "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Rarely", "BusinessTravel_Travel_Frequently",
    "EducationField_Human Resources", "EducationField_Life Sciences", "EducationField_Marketing",
    "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree"
]


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


# ⚡ Route Prediksi
@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    model = load_model()
    prediction, probability = None, None

    education_options = [
        "Human Resources", "Life Sciences", "Marketing",
        "Medical", "Other", "Technical Degree"
    ]

    if request.method == "GET":
        features = {
            "EducationField": random.choice(education_options),
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
        form = request.form
        features = {f: form.get(f) for f in FEATURES}
        encoded = {}

        # ===== Encoding =====
        encoded["OverTime_Yes"] = 1.0 if features["OverTime"] == "Yes" else 0.0

        for ms in ["Single", "Married", "Divorced"]:
            encoded[f"MaritalStatus_{ms}"] = 1.0 if features["MaritalStatus"] == ms else 0.0

        roles = [
            "Human Resources", "Healthcare Representative", "Research Scientist",
            "Sales Executive", "Manager", "Laboratory Technician",
            "Research Director", "Manufacturing Director", "Sales Representative"
        ]
        for role in roles:
            encoded[f"JobRole_{role}"] = 1.0 if features["JobRole"] == role else 0.0

        travels = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        for bt in travels:
            encoded[f"BusinessTravel_{bt}"] = 1.0 if features["BusinessTravel"] == bt else 0.0

        educations = [
            "Human Resources", "Life Sciences", "Marketing",
            "Medical", "Other", "Technical Degree"
        ]
        for ef in educations:
            encoded[f"EducationField_{ef}"] = 1.0 if features["EducationField"] == ef else 0.0

        encoded["StockOptionLevel"] = float(features["StockOptionLevel"])
        encoded["JobLevel"] = float(features["JobLevel"])
        encoded["EnvironmentSatisfaction"] = float(features["EnvironmentSatisfaction"])
        encoded["JobInvolvement"] = float(features["JobInvolvement"])
        encoded["JobSatisfaction"] = float(features["JobSatisfaction"])

        df = pd.DataFrame([{f: encoded.get(f, 0) for f in ENCODED_FEATURES}])

        # 🧩 Urutkan kolom sesuai urutan model agar tidak mismatch
        try:
            model_features = model.get_booster().feature_names
            df = df.reindex(columns=model_features, fill_value=0)
        except Exception:
            pass

        # ===== Prediksi =====
        try:
            pred_label = model.predict(df)[0]
            pred_proba = model.predict_proba(df)[0]

            prob_yes = float(pred_proba[1]) * 100
            prob_no = float(pred_proba[0]) * 100

            prediction = "Yes" if pred_label == 1 else "No"
            probability = f"{prob_yes:.2f}%" if pred_label == 1 else f"{prob_no:.2f}%"
        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = None

    return render_template(
        "predict_view.html",
        features=features,
        prediction=prediction,
        probability=probability,
        education_options=education_options
    )


# 🔹 API Predict
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    model = load_model()

    encoded = {}
    encoded["OverTime_Yes"] = 1.0 if data.get("OverTime") == "Yes" else 0.0

    for ms in ["Single", "Married", "Divorced"]:
        encoded[f"MaritalStatus_{ms}"] = 1.0 if data.get("MaritalStatus") == ms else 0.0

    roles = [
        "Human Resources", "Healthcare Representative", "Research Scientist",
        "Sales Executive", "Manager", "Laboratory Technician",
        "Research Director", "Manufacturing Director", "Sales Representative"
    ]
    for role in roles:
        encoded[f"JobRole_{role}"] = 1.0 if data.get("JobRole") == role else 0.0

    travels = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    for bt in travels:
        encoded[f"BusinessTravel_{bt}"] = 1.0 if data.get("BusinessTravel") == bt else 0.0

    educations = [
        "Human Resources", "Life Sciences", "Marketing",
        "Medical", "Other", "Technical Degree"
    ]
    for ef in educations:
        encoded[f"EducationField_{ef}"] = 1.0 if data.get("EducationField") == ef else 0.0

    encoded["StockOptionLevel"] = float(data.get("StockOptionLevel", 1))
    encoded["JobLevel"] = float(data.get("JobLevel", 2))
    encoded["EnvironmentSatisfaction"] = float(data.get("EnvironmentSatisfaction", 3))
    encoded["JobInvolvement"] = float(data.get("JobInvolvement", 3))
    encoded["JobSatisfaction"] = float(data.get("JobSatisfaction", 3))

    df = pd.DataFrame([{f: encoded.get(f, 0) for f in ENCODED_FEATURES}])

    # 🧩 Pastikan urutan kolom sesuai model
    try:
        model_features = model.get_booster().feature_names
        df = df.reindex(columns=model_features, fill_value=0)
    except Exception:
        pass

    pred_label = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0]
    prob_yes = float(pred_proba[1]) * 100
    prob_no = float(pred_proba[0]) * 100

    prediction = "Yes" if pred_label == 1 else "No"
    probability = f"{prob_yes:.2f}%" if pred_label == 1 else f"{prob_no:.2f}%"

    return jsonify({
        "prediction": prediction,
        "confidence": probability,
        "features": data
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
