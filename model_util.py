import os
import joblib
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_model():
    """
    Mengunduh model dari DagsHub MLflow Registry jika model lokal tidak tersedia.
    """
    dagshub_uri = "https://dagshub.com/NauraaSalsabila/DSP_ATTRITION.mlflow"
    mlflow.set_tracking_uri(dagshub_uri)

    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    # Validasi kredensial
    if not username or not token:
        print("⚠️ DagsHub credentials tidak ditemukan di environment variables. "
              "Gunakan model lokal sebagai fallback.")
        return None

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    model_uri = "models:/xgb-attrition-model/1"
    local_path = "model/xgb_attrition_model.pkl"

    print(f"📥 Mengunduh model dari MLflow Registry: {model_uri} ...")
    try:
        model = mlflow.xgboost.load_model(model_uri)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        joblib.dump(model, local_path)
        print(f"✅ Model XGBoost berhasil disimpan di: {local_path}")
        return model
    except Exception as e:
        print(f"❌ Gagal mengunduh model dari DagsHub: {e}")
        return None


def load_model():
    """
    Memuat model dari file lokal jika tersedia, atau mencoba unduh dari DagsHub jika tidak.
    """
    local_path = "model/xgb_attrition_model.pkl"
    abs_path = os.path.abspath(local_path)

    print(f"🔍 Memeriksa model lokal di: {abs_path}")

    # Coba pakai model lokal dulu
    if os.path.exists(local_path):
        try:
            model = joblib.load(local_path)
            print(f"✅ Model XGBoost berhasil dimuat dari: {local_path}")
            return model
        except Exception as e:
            print(f"❌ Gagal memuat model lokal: {e}")

    # Kalau model lokal tidak ada, coba unduh dari DagsHub
    print("⚠️ Model lokal tidak ditemukan. Mencoba unduh dari DagsHub...")
    model = get_model()

    if model is None:
        print("❌ Tidak dapat memuat model dari sumber mana pun.")
        return None

    return model
