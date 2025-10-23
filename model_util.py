import os
import joblib
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Konfigurasi DagsHub ===
DAGSHUB_URI = "https://dagshub.com/NauraaSalsabila/DSP_ATTRITION.mlflow"
MODEL_NAME = "xgb-attrition-model-encoded"  
MODEL_VERSION = "2" 
LOCAL_MODEL_PATH = "model/xgb_attrition_model_encoded.pkl"


def get_model():
    """
    Mengunduh model terbaru dari DagsHub MLflow Model Registry.
    Jika berhasil, model juga disimpan secara lokal agar bisa digunakan offline.
    """
    mlflow.set_tracking_uri(DAGSHUB_URI)

    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    # Validasi kredensial
    if not username or not token:
        print("⚠️ DagsHub credentials tidak ditemukan. "
              "Pastikan DAGSHUB_USERNAME dan DAGSHUB_TOKEN ada di .env.")
        return None

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

    print(f"📥 Mengunduh model dari MLflow Registry: {model_uri} ...")
    try:
        model = mlflow.xgboost.load_model(model_uri)

        # Simpan lokal
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        joblib.dump(model, LOCAL_MODEL_PATH)
        print(f"✅ Model berhasil diunduh dan disimpan di: {LOCAL_MODEL_PATH}")
        return model

    except Exception as e:
        print(f"❌ Gagal mengunduh model dari DagsHub: {e}")
        return None


def load_model():
    """
    Memuat model XGBoost:
    - Pertama dari file lokal (offline mode)
    - Jika tidak ada, otomatis download dari DagsHub (online)
    """
    abs_path = os.path.abspath(LOCAL_MODEL_PATH)
    print(f"🔍 Memeriksa model lokal di: {abs_path}")

    # Coba load model lokal dulu
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            print(f"✅ Model berhasil dimuat dari lokal: {LOCAL_MODEL_PATH}")
            return model
        except Exception as e:
            print(f"⚠️ Gagal memuat model lokal: {e}")

    # Kalau gagal atau belum ada file, download dari DagsHub
    print("🌐 Model lokal tidak ditemukan. Mencoba unduh dari DagsHub...")
    model = get_model()

    if model is None:
        print("❌ Tidak dapat memuat model dari sumber mana pun.")
        return None

    return model
