import os
import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import load_dataset

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from tensorflow.keras import (
    layers,
    Model
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(
    BASE_DIR,
    "laptops.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model_artifacts"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ai_recommender.pkl"
)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _parse_ram(x):
    nums = re.findall(r"\d+", str(x))
    return float(nums[0]) if nums else 8.0

def _parse_storage(x):
    x = str(x).upper()
    nums = re.findall(r"\d+", x)
    val = float(nums[0]) if nums else 256.0
    if "TB" in x:
        val *= 1024
    return val

def _parse_screen(x):
    nums = re.findall(r"\d+\.?\d*", str(x))
    return float(nums[0]) if nums else 15.6

def _extract_title_from_link(link):
    if pd.isna(link):
        return "Laptop"
    link = str(link)
    match = re.search(r"/([^/]+)-laptop", link.lower())
    if match:
        return match.group(1).replace("-", " ").title()
    return "Premium Laptop"

# ─────────────────────────────────────────────────────────────
# TRAINING DATASET
# ─────────────────────────────────────────────────────────────

def load_training_dataset():
    print("[AI] Loading training dataset from HuggingFace...")
    dataset = load_dataset("Ammok/laptop_price_prediction")
    df = pd.DataFrame(dataset["train"])
    df.columns = df.columns.str.strip()

    rename_dict = {
        "Price": "price",
        "CPU": "cpu",
        "RAM": "ram",
        "Storage": "storage",
        "Screen Size": "screen",
        "Brand": "brand",
        "GPU": "gpu"
    }
    df = df.rename(columns=rename_dict)

    for col in ["brand", "gpu"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["ram_gb"]     = df["ram"].apply(_parse_ram)
    df["storage_gb"] = df["storage"].apply(_parse_storage)
    df["screen_in"]  = df["screen"].apply(_parse_screen)
    df["price"]      = pd.to_numeric(df["price"], errors="coerce").fillna(3000)

    encoders = {}
    for col in ["brand", "cpu", "gpu"]:
        enc = LabelEncoder()
        df[f"{col}_encoded"] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc

    return df, encoders

# ─────────────────────────────────────────────────────────────
# LOCAL DATASET
# ─────────────────────────────────────────────────────────────

def load_local_dataset(encoders):
    print("[AI] Loading laptops.csv...")
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    # title
    if "title" not in df.columns:
        if "link" in df.columns:
            df["title"] = df["link"].apply(_extract_title_from_link)
        else:
            df["title"] = "Laptop"

    # missing text columns
    for col in ["brand", "gpu"]:
        if col not in df.columns:
            df[col] = "Unknown"

    if "os_version" not in df.columns:
        for alt in ["os", "OS", "operating_system", "Operating System"]:
            if alt in df.columns:
                df["os_version"] = df[alt]
                break
        else:
            df["os_version"] = "Unknown"

    if "rating" not in df.columns:
        for alt in ["Rating", "rate", "stars", "Stars"]:
            if alt in df.columns:
                df["rating"] = df[alt]
                break
        else:
            df["rating"] = 0

    # numeric: ram
    if "ram_gb" not in df.columns:
        if "ram" in df.columns:
            df["ram_gb"] = df["ram"].apply(_parse_ram)
        else:
            df["ram_gb"] = 8

    # numeric: storage
    if "storage_gb" not in df.columns:
        if "storage" in df.columns:
            df["storage_gb"] = df["storage"].apply(_parse_storage)
        else:
            df["storage_gb"] = 256

    # numeric: screen
    if "screen_in" not in df.columns:
        if "screen" in df.columns:
            df["screen_in"] = df["screen"].apply(_parse_screen)
        else:
            df["screen_in"] = 15.6

    # price
    df["price"] = pd.to_numeric(
        df.get("price", 0), errors="coerce"
    ).fillna(3000)

    # encode text columns
    for col in ["brand", "cpu", "gpu"]:
        enc = encoders[col]
        mapping = {cls: idx for idx, cls in enumerate(enc.classes_)}
        if col not in df.columns:
            df[col] = "Unknown"
        df[f"{col}_encoded"] = df[col].apply(lambda x: mapping.get(str(x), 0))

    # image
    if "image_main" not in df.columns:
        if "image" in df.columns:
            df["image_main"] = df["image"]
        else:
            df["image_main"] = ""

    # link
    if "link" not in df.columns:
        df["link"] = ""

    needed = [
        "title",
        "price",
        "cpu",
        "ram",
        "storage",
        "screen",
        "brand",
        "gpu",
        "link",
        "image_main",
        "os_version",   
        "rating",       
        "brand_encoded",
        "cpu_encoded",
        "gpu_encoded",
        "ram_gb",
        "storage_gb",
        "screen_in",
    ]

    for col in needed:
        if col not in df.columns:
            df[col] = ""

    return df[needed].copy()

# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────

FEATURES = [
    "brand_encoded",
    "cpu_encoded",
    "gpu_encoded",
    "ram_gb",
    "storage_gb",
    "screen_in",
    "price"
]

# ─────────────────────────────────────────────────────────────
# AUTOENCODER MODEL
# ─────────────────────────────────────────────────────────────

def build_autoencoder(input_dim):
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    embedding = layers.Dense(32, activation="linear", name="embedding")(x)

    x = layers.Dense(64, activation="relu")(embedding)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inputs, outputs)
    encoder     = Model(inputs, embedding)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder

# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────

def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df_train, encoders = load_training_dataset()

    X_train = df_train[FEATURES].values.astype(np.float32)
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    autoencoder, encoder = build_autoencoder(input_dim=X_train.shape[1])

    print("[AI] Training AutoEncoder...")
    autoencoder.fit(
        X_train, X_train,
        epochs=30,
        batch_size=32,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    df_local = load_local_dataset(encoders)

    X_local = df_local[FEATURES].values.astype(np.float32)
    X_local = scaler.transform(X_local)

    embeddings = encoder.predict(X_local)

    artifact = {
        "encoder":    encoder,
        "scaler":     scaler,
        "encoders":   encoders,
        "df":         df_local,
        "embeddings": embeddings,
        "features":   FEATURES
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"[AI] Trained on {len(df_train)} internet laptops")
    print(f"[AI] Serving {len(df_local)} local laptops")

    return artifact

# ─────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────

_artifact = None

def get_artifact():
    global _artifact
    if _artifact is None:
        if os.path.exists(MODEL_PATH):
            _artifact = joblib.load(MODEL_PATH)
            print("[AI] Loaded cached model.")
        else:
            _artifact = train_model()
    return _artifact

# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def get_all_laptops():
    df   = get_artifact()["df"]
    data = df.to_dict("records")
    for i, row in enumerate(data):
        row["id"] = i
    return data

def get_laptop(idx):
    df = get_artifact()["df"]
    if idx < 0 or idx >= len(df):
        raise IndexError(f"Invalid laptop id: {idx}")
    row       = df.iloc[idx].to_dict()
    row["id"] = idx
    return row

# ─────────────────────────────────────────────────────────────
# AI RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────

def get_recommendations(idx, top_n=6):
    art        = get_artifact()
    df         = art["df"]
    embeddings = art["embeddings"]

    if idx < 0 or idx >= len(df):
        return []

    target = embeddings[idx].reshape(1, -1)
    sims   = cosine_similarity(target, embeddings)[0]
    sims[idx] = -1  

    ranked  = np.argsort(sims)[::-1][:top_n]
    results = []

    for i in ranked:
        row          = df.iloc[i].to_dict()
        row["id"]    = int(i)
        row["score"] = float(sims[i])
        results.append(row)

    return results

# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    laptops = get_all_laptops()
    print("\nTOTAL:", len(laptops))

    if laptops:
        print("\nFIRST LAPTOP:\n")
        first = laptops[0]
        for k in ["title", "cpu", "ram", "storage", "price", "os_version", "rating"]:
            print(f"{k}: {first.get(k)}")

        print("\nAI RECOMMENDATIONS:\n")
        recs = get_recommendations(0, top_n=5)
        for r in recs:
            print(f"[{r['score']:.4f}] {r['title']}")