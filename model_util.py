import os
import joblib
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_model():
    dagshub_uri = "https://dagshub.com/NauraaSalsabila/DSP_ATTRITION.mlflow"
    mlflow.set_tracking_uri(dagshub_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    # URI model dari DagsHub (lihat di tab Models)
    model_uri = "models:/xgb-attrition-model/1"  
    local_path = "model/xgb_attrition_model.pkl"

    print(f"📥 Mengunduh model dari MLflow Registry: {model_uri} ...")
    try:
        model = mlflow.xgboost.load_model(model_uri)
        joblib.dump(model, local_path)
        print(f"✅ Model XGBoost berhasil disimpan di: {local_path}")
        return model
    except Exception as e:
        print(f"❌ Gagal mengunduh model dari DagsHub: {e}")
        return None


def load_model():
    local_path = "model/xgb_attrition_model.pkl"

    if not os.path.exists(local_path):
        print("⚠️ Model lokal belum ditemukan. Mengunduh dari DagsHub...")
        model = get_model()
        if model is None:
            print("❌ Tidak bisa memuat model XGBoost.")
            return None
        return model

    try:
        model = joblib.load(local_path)
        print(f"✅ Model XGBoost berhasil dimuat dari: {local_path}")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model lokal: {e}")
        return None
