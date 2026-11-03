# User-Level Social Text Classification & Explainability

Research prototype: user-level text classification with fine-tuned BERT, post-level KernelSHAP, and grounded Groq-generated explanations. This is not a clinical tool; labels are dataset-defined.

## Structure
- `src/data.py`: parse records, select recent posts, create datasets/splits
- `src/model.py`: BERT user-level classifier
- `src/train_frozen.py`: frozen BERT embeddings + Logistic Regression baseline
- `src/train.py`: fine-tune/evaluate/save model
- `src/compare_models.py`: compare saved evaluation metrics
- `app.py`: Streamlit presentation frontend
- `src/explain.py`: prediction, SHAP, and Groq explanation
- `data/`: local dataset (not committed)
- `models/`: saved model/tokenizer (not committed)
- `outputs/`: metrics

## Setup
Python 3.10+:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env`, then set `GROQ_API_KEY`. Never commit `.env`.

Place `user_level_text_dataset_clean.csv` in `data/`. Required columns: `label`, `posts`. `posts` must be a JSON list of strings in each CSV cell. Optional: `timestamps`, `user_id`.

Train the frozen BERT baseline:
```bash
python -m src.train_frozen --data data/user_level_text_dataset_clean.csv
```

Train the fine-tuned BERT model:
```bash
python -m src.train --data data/user_level_text_dataset_clean.csv
```

Compare saved metrics:
```bash
python -m src.compare_models
```
Explain a row:
```bash
python -m src.explain --data data/user_level_text_dataset_clean.csv --row-index 0
```

Do not commit raw datasets, model weights, API keys, or private social-media text. Review dataset terms before redistribution.


## Streamlit presentation frontend

After training the fine-tuned model and configuring `.env`, launch:

```bash
streamlit run app.py
```

The browser interface includes a project overview, architecture, user-row selector, BERT + SHAP + Groq explanation, and a contribution table/chart. Keep the dataset local and do not expose raw social-media text in public demos.
