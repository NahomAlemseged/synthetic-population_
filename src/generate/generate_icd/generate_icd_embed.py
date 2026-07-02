import pandas as pd
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time
from scipy.spatial.distance import jensenshannon

# --------------------------
# CONFIG
# --------------------------
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)

train_path = params["generate_icd"]["input"][0]
pop_path = params["generate_icd"]["input"][1]

TRAIN_PATH = Path(train_path)
POP_PATH = Path(pop_path)

OUTPUT_PATH = Path(params["generate_icd"]["output"])
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_PATH / "icd_embedding_results.csv"

TRAIN_SAMPLE_SIZE = 100000
TOP_K = 3


# --------------------------
# MODEL
# --------------------------
class ICDEmbeddingModel(nn.Module):
    def __init__(self, input_dim, icd_vocab_size, embed_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

        self.icd_embeddings = nn.Embedding(icd_vocab_size, embed_dim)

    def encode(self, x):
        return self.encoder(x)


# --------------------------
# GENERATOR
# --------------------------
class ICDGenerator:

    def __init__(self, train_path, pop_path):
        self.train_path = train_path
        self.pop_path = pop_path

    def load_data(self):
        df_train = pd.read_csv(self.train_path, dtype=str)
        df_pop = pd.read_csv(self.pop_path, dtype=str)

        print(f"📊 Train: {df_train.shape}")
        print(f"📊 Pop: {df_pop.shape}")

        return df_train, df_pop

    def sample(self, df):
        return df.sample(min(TRAIN_SAMPLE_SIZE, len(df)), random_state=42)

    def preprocess(self, df, features, target):
        df = df[features + [target]].dropna()

        encoders = {}
        for col in features + [target]:
            df[col], _ = pd.factorize(df[col])

        return df

    def train(self, df, features, target):

        X = torch.tensor(df[features].values, dtype=torch.float32)
        y = torch.tensor(df[target].values, dtype=torch.long)

        vocab_size = len(torch.unique(y))

        model = ICDEmbeddingModel(X.shape[1], vocab_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("🚀 Training embedding model...")

        for epoch in range(5):

            patient_vec = model.encode(X)
            icd_vecs = model.icd_embeddings.weight

            logits = torch.matmul(patient_vec, icd_vecs.T)

            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        return model, vocab_size

    # --------------------------
    # PREDICTION
    # --------------------------
    def predict(self, model, df, features):

        X = torch.tensor(df[features].values, dtype=torch.float32)

        with torch.no_grad():
            patient_vec = model.encode(X)
            icd_vecs = model.icd_embeddings.weight

            logits = torch.matmul(patient_vec, icd_vecs.T)

            top1 = torch.argmax(logits, dim=1)
            topk = torch.topk(logits, k=TOP_K, dim=1).indices

        return top1.numpy(), topk.numpy()

    # --------------------------
    # ACCURACY
    # --------------------------
    def accuracy(self, y_true, y_pred):

        return np.mean(y_true == y_pred)

    def topk_accuracy(self, y_true, topk_preds):

        correct = 0

        for i, true in enumerate(y_true):
            if true in topk_preds[i]:
                correct += 1

        return correct / len(y_true)

    # --------------------------
    # JENSEN-SHANNON SIMILARITY
    # --------------------------
    def js_similarity(self, y_true, y_pred):

        true_dist = np.bincount(y_true) + 1e-9
        pred_dist = np.bincount(y_pred) + 1e-9

        # normalize
        true_dist = true_dist / true_dist.sum()
        pred_dist = pred_dist / pred_dist.sum()

        return 1 - jensenshannon(true_dist, pred_dist)


# --------------------------
# MAIN
# --------------------------
def main():

    print("⚙️ ICD Embedding Pipeline Starting...")

    start = time.time()

    features = [
        "SEX_CODE",
        "PAT_AGE",
        "RACE",
        "ETHNICITY",
        "PAT_ZIP",
        "PAT_COUNTY",
        "PUBLIC_HEALTH_REGION",
        "APR_MDC"
    ]

    target = "PRINC_DIAG_CODE"

    gen = ICDGenerator(TRAIN_PATH, POP_PATH)

    # Load
    df_train, df_pop = gen.load_data()

    # Sample
    df_train = gen.sample(df_train)

    # Preprocess
    df_train = gen.preprocess(df_train, features, target)

    # Train model
    model, vocab = gen.train(df_train, features, target)

    # --------------------------
    # REAL LABELS (for evaluation)
    # --------------------------
    y_true = pd.factorize(df_pop[target])[0]

    # Predict
    top1, topk = gen.predict(model, df_pop, features)

    # --------------------------
    # METRICS
    # --------------------------
    acc = gen.accuracy(y_true, top1)
    topk_acc = gen.topk_accuracy(y_true, topk)
    js_sim = gen.js_similarity(y_true, top1)

    print("\n📊 FINAL RESULTS")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Top-{TOP_K} Accuracy: {topk_acc:.4f}")
    print(f"Jensen-Shannon Similarity: {js_sim:.4f}")

    # save
    df_out = df_pop.copy()
    df_out["ICD_TOP1"] = top1

    df_out.to_csv(OUTPUT_CSV, index=False)

    end = time.time()

    print(f"\n💾 Saved: {OUTPUT_CSV}")
    print(f"⏱️ Time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
