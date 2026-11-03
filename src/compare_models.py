"""Display metrics produced by the frozen and fine-tuned training scripts."""
import json
from pathlib import Path
from .config import OUTPUT_DIR


def main():
    files = {
        "Frozen BERT + Logistic Regression": OUTPUT_DIR / "frozen_bert_results.json",
        "Fine-tuned BERT": OUTPUT_DIR / "results.json",
    }
    rows = []
    for name, path in files.items():
        if not path.exists():
            print(f"Missing {path}. Run the corresponding training script first.")
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": name,
            "test_accuracy": metrics.get("test_accuracy"),
            "test_f1": metrics.get("test_f1"),
        })
    if rows:
        print(f"{'Model':38} {'Test accuracy':>14} {'Test F1':>10}")
        print("-" * 65)
        for row in rows:
            acc = "N/A" if row["test_accuracy"] is None else f'{row["test_accuracy"]:.4f}'
            f1 = "N/A" if row["test_f1"] is None else f'{row["test_f1"]:.4f}'
            print(f'{row["model"]:38} {acc:>14} {f1:>10}')


if __name__ == "__main__":
    main()
