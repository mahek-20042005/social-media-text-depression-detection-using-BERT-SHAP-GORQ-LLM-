from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "fine_tuned_bert_model"
OUTPUT_DIR = ROOT / "outputs"
BASE_MODEL = os.getenv("BASE_MODEL_NAME", "bert-base-uncased")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_POSTS = 10
MAX_TOKENS = 128
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 1
SEED = 42
