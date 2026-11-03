import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from dotenv import load_dotenv
from groq import Groq
from transformers import BertTokenizerFast

from .config import BASE_MODEL, GROQ_MODEL, MODEL_DIR, MAX_POSTS, MAX_TOKENS
from .model import UserBERTClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UserExplainer:
    """Predict and explain a set of posts belonging to one user."""

    def __init__(self):
        load_dotenv(PROJECT_ROOT / ".env")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer.json is present in this project's fine-tuned model directory.
        self.tokenizer = BertTokenizerFast.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        self.model = UserBERTClassifier().to(self.device)
        checkpoint_path = MODEL_DIR / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.client = Groq(api_key=api_key) if api_key else None
        self.llm = os.getenv("GROQ_MODEL", GROQ_MODEL).strip()

    @torch.no_grad()
    def predict(self, posts):
        """Return positive-class probability for one user's list of posts."""
        posts = [str(post).strip() for post in posts if str(post).strip()]
        posts = posts[-MAX_POSTS:]
        n_posts = len(posts)

        if n_posts == 0:
            raise ValueError("Enter at least one non-empty post.")

        padded_posts = posts + [""] * (MAX_POSTS - n_posts)
        encoded = self.tokenizer(
            padded_posts,
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
        post_mask = torch.tensor(
            [[1.0] * n_posts + [0.0] * (MAX_POSTS - n_posts)],
            dtype=torch.float32,
            device=self.device,
        )

        logits = self.model(input_ids, attention_mask, post_mask)
        return float(torch.softmax(logits, dim=1)[0, 1].cpu())

    def explain(self, posts, nsamples=100, top_k=3):
        """Compute post-level KernelSHAP values and optionally request Groq text."""
        posts = [str(post).strip() for post in posts if str(post).strip()]
        posts = posts[-MAX_POSTS:]
        n_posts = len(posts)

        if n_posts == 0:
            raise ValueError("Enter at least one non-empty post.")

        padded_posts = posts + [""] * (MAX_POSTS - n_posts)
        observed_mask = np.array([1.0] * n_posts + [0.0] * (MAX_POSTS - n_posts))

        positive_probability = self.predict(posts)
        predicted_class = "Positive" if positive_probability >= 0.5 else "Negative"

        def predict_masks(masks):
            predictions = []
            for row_mask in np.asarray(masks):
                included_posts = [
                    padded_posts[i]
                    for i in range(n_posts)
                    if row_mask[i] >= 0.5
                ]
                # Empty subsets are represented by the model's zero-post baseline.
                if not included_posts:
                    predictions.append(0.0)
                else:
                    predictions.append(self.predict(included_posts))
            return np.asarray(predictions)

        explainer = shap.KernelExplainer(
            predict_masks,
            np.zeros((1, MAX_POSTS)),
        )
        shap_values = np.asarray(
            explainer.shap_values(
                observed_mask.reshape(1, -1),
                nsamples=int(nsamples),
            )
        ).reshape(-1)

        shap_table = pd.DataFrame(
            {
                "post_number": range(1, MAX_POSTS + 1),
                "post": padded_posts,
                "shap_value": shap_values,
                "real": observed_mask.astype(bool),
            }
        )
        shap_table = shap_table[shap_table["real"]].copy()
        shap_table["direction"] = np.where(
            shap_table["shap_value"] >= 0,
            "Toward Positive",
            "Away from Positive",
        )
        shap_table = shap_table.sort_values(
            "shap_value",
            key=lambda values: values.abs(),
            ascending=False,
        )

        top_positive = shap_table[shap_table["shap_value"] > 0].nlargest(
            top_k, "shap_value"
        )
        top_negative = shap_table[shap_table["shap_value"] < 0].nsmallest(
            top_k, "shap_value"
        )

        def evidence(frame):
            if frame.empty:
                return "None"
            return "\n".join(
                f"Post {int(row.post_number)} (SHAP {row.shap_value:+.4f}): {row.post}"
                for row in frame.itertuples()
            )

        if self.client is None:
            explanation = (
                "Groq explanation is unavailable because GROQ_API_KEY is not set "
                "in the project's .env file. The prediction and SHAP results were "
                "computed successfully."
            )
        else:
            prompt = f"""Explain this research model output, not a person's medical condition.

Prediction: {predicted_class}
Positive-class probability: {positive_probability:.4f}

Positive SHAP evidence:
{evidence(top_positive)}

Negative SHAP evidence:
{evidence(top_negative)}

Rules:
- This is not a clinical tool; do not diagnose or infer symptoms, motives, or mental states.
- Use only the supplied posts.
- Positive/negative SHAP indicates contribution toward/away from the positive class
  relative to the reference; SHAP values are not direct probability changes.
- Do not claim a phrase caused the output or invent the model's mechanism.
- Distinguish dataset labels from verified real-world outcomes.
- If the text does not clearly contain relevant language, say so.
Return these sections: Prediction, Supporting evidence, Counter-evidence,
Grounded interpretation, Caveat."""

            response = self.client.chat.completions.create(
                model=self.llm,
                temperature=0.2,
                max_tokens=800,
                messages=[
                    {
                        "role": "system",
                        "content": "Provide cautious, evidence-grounded model explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            explanation = response.choices[0].message.content

        return {
            "predicted_class": predicted_class,
            "positive_probability": positive_probability,
            "shap_table": shap_table,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "explanation": explanation,
            "device": str(self.device),
            "posts_used": n_posts,
        }
