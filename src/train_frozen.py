"""Frozen BERT embeddings + Logistic Regression baseline."""
import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

try:
    from .config import BASE_MODEL, OUTPUT_DIR, MODEL_DIR, MAX_POSTS, MAX_TOKENS, SEED
    from .data import load_data, split_data
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import BASE_MODEL, OUTPUT_DIR, MODEL_DIR, MAX_POSTS, MAX_TOKENS, SEED
    from src.data import load_data, split_data


@torch.no_grad()
def embed_posts(posts, tokenizer, bert, device, batch_size=16):
    """Return one 768-D mean-pooled embedding per post."""
    vectors = []
    bert.eval()
    for start in tqdm(range(0, len(posts), batch_size), desc="Embedding posts"):
        batch_posts = posts[start:start + batch_size]
        enc = tokenizer(batch_posts, padding="max_length", truncation=True,
                        max_length=MAX_TOKENS, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = bert(input_ids=ids, attention_mask=mask).last_hidden_state
        token_mask = mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1e-9)
        vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def user_embeddings(df, tokenizer, bert, device):
    """Average each user's post embeddings into a single user vector."""
    all_posts, owners = [], []
    for idx, posts in enumerate(df["selected_posts"]):
        for post in posts[-MAX_POSTS:]:
            all_posts.append(str(post))
            owners.append(idx)
    post_vectors = embed_posts(all_posts, tokenizer, bert, device)
    X = np.zeros((len(df), post_vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(len(df), dtype=np.int32)
    for owner, vec in zip(owners, post_vectors):
        X[owner] += vec
        counts[owner] += 1
    X /= np.maximum(counts[:, None], 1)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/user_level_text_dataset_clean.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data(args.data)
    train_df, val_df, test_df = split_data(df)

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    bert = BertModel.from_pretrained(BASE_MODEL).to(device)
    for parameter in bert.parameters():
        parameter.requires_grad = False

    # Fit/transform each split with the same frozen pretrained encoder.
    X_train = user_embeddings(train_df, tokenizer, bert, device)
    X_val = user_embeddings(val_df, tokenizer, bert, device)
    X_test = user_embeddings(test_df, tokenizer, bert, device)
    y_train, y_val, y_test = train_df.label.to_numpy(), val_df.label.to_numpy(), test_df.label.to_numpy()

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    test_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "method": "Frozen BERT mean-pooled post embeddings + Logistic Regression",
        "base_model": BASE_MODEL,
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frozen_dir = MODEL_DIR.parent / "frozen_bert"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(clf, frozen_dir / "logistic_regression.joblib")
    tokenizer.save_pretrained(frozen_dir / "tokenizer")
    metrics_path = OUTPUT_DIR / "frozen_bert_results.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(y_test, test_pred, zero_division=0))
    print(json.dumps(metrics, indent=2))
    print(f"Saved Logistic Regression classifier: {frozen_dir / 'logistic_regression.joblib'}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
