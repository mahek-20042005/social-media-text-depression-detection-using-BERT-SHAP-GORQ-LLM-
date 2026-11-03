import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MODEL_DIR, MAX_POSTS
from src.explain import UserExplainer


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="Social Text Explainability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .main-title {font-size: 2.25rem; font-weight: 750; letter-spacing: -0.04em;}
    .subtle {color: #9CA3AF; font-size: 0.95rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧠 Social Text Classification & Explainability</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Research prototype · Fine-tuned BERT · KernelSHAP · Groq LLM</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------- Load saved evaluation results ----------
def read_json(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


fine = read_json("fine_tune_bert_results.json")
frozen = read_json("frozen_bert_results.json")

# Support alternate filename used in earlier project versions.
if not fine:
    fine = read_json("fine_tuned_bert_results.json")
if not frozen:
    frozen = read_json("frozen_bert_results.json")


@st.cache_resource(show_spinner="Loading fine-tuned BERT and tokenizer...")
def get_explainer():
    return UserExplainer()


with st.sidebar:
    st.header("Project navigation")
    st.caption("Explore model evaluation or test your own text.")
    st.markdown("---")
    st.markdown("**Model:** `bert-base-uncased`")
    st.markdown(f"**Maximum posts per user:** {MAX_POSTS}")
    st.markdown("**Tokens per post:** 128")
    st.markdown("---")
    st.caption("The output represents dataset-defined classes, not a medical diagnosis.")

overview_tab, comparison_tab, prediction_tab, methodology_tab = st.tabs(
    ["🏠 Overview", "📊 Model comparison", "🔎 Live prediction", "📚 Methodology"]
)

# ---------- Overview ----------
with overview_tab:
    st.subheader("Project at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classification models", "2")
    c2.metric("Fine-tuned test accuracy", f'{fine.get("test_accuracy", 0):.2%}' if fine else "N/A")
    c3.metric("Frozen BERT test accuracy", f'{frozen.get("test_accuracy", 0):.2%}' if frozen else "N/A")
    c4.metric("Posts per user", str(MAX_POSTS))

    st.markdown("### End-to-end architecture")
    stages = st.columns(5)
    stage_labels = [
        ("01 · Input", "User's social posts"),
        ("02 · NLP", "Tokenization and BERT"),
        ("03 · Prediction", "Dataset-defined class"),
        ("04 · Explainability", "Post-level SHAP values"),
        ("05 · Interpretation", "Groq LLM summary"),
    ]
    for col, (heading, detail) in zip(stages, stage_labels):
        with col:
            st.markdown(f"**{heading}**")
            st.caption(detail)

    st.info(
        "Use the Live prediction tab to enter one or more posts. The application "
        "treats the entered posts as belonging to one user and sends them to the "
        "fine-tuned BERT model."
    )

    st.markdown("### Research scope")
    st.write(
        "This prototype studies user-level text classification using labels supplied "
        "by the dataset. It is not a clinical screening, diagnostic, or treatment tool. "
        "Model probabilities and SHAP contributions should be interpreted only as "
        "outputs of this trained classifier."
    )


# ---------- Model comparison ----------
with comparison_tab:
    st.subheader("Evaluation results")
    st.caption(
        "Metrics are loaded from saved JSON files. No model training is performed."
    )

    if not fine or not frozen:
        st.warning(
            "Could not find both result files. Place fine_tune_bert_results.json "
            "and frozen_bert_results.json inside the outputs/ folder."
        )

    else:
        # Extract metrics
        frozen_accuracy = frozen.get("test_accuracy", 0)
        frozen_f1 = frozen.get("test_f1", 0)

        fine_accuracy = fine.get("test_accuracy", 0)
        fine_f1 = fine.get("test_positive_f1", fine.get("test_f1", 0))

        # ---------- Summary metrics ----------
        a, b = st.columns(2)

        a.metric(
            "Fine-tuned BERT · Test accuracy",
            f"{fine_accuracy:.2%}",
            f"{(fine_accuracy - frozen_accuracy) * 100:+.2f} percentage points"
        )

        b.metric(
            "Fine-tuned BERT · Test F1-score",
            f"{fine_f1:.2%}",
            f"{(fine_f1 - frozen_f1) * 100:+.2f} percentage points"
        )

        st.divider()

        # ---------- 1. Test-set comparison ----------
        st.markdown("### 1. Test-set performance comparison")
        st.caption(
            "Grouped bars compare the two models independently for each metric. "
            "The vertical axis starts at zero."
        )

        test_comparison = pd.DataFrame({
            "Metric": ["Accuracy", "F1-score"],
            "Frozen BERT + Logistic Regression": [
                frozen_accuracy,
                frozen_f1
            ],
            "Fine-tuned BERT": [
                fine_accuracy,
                fine_f1
            ]
        }).set_index("Metric")

        st.bar_chart(
            test_comparison,
            y_label="Score (0–1)",
            stack=False
        )

        # Exact values below the chart
        st.dataframe(
            (test_comparison * 100).round(2).rename_axis("Metric"),
            use_container_width=True
        )

        st.caption(
            "All values are percentages in the table. Higher accuracy and F1 "
            "indicate better performance on this test split."
        )

        st.divider()

        # ---------- 2. Validation vs test: accuracy ----------
        st.markdown("### 2. Accuracy: validation vs test")
        st.caption(
            "Compares each model's accuracy on the validation and test splits."
        )

        accuracy_comparison = pd.DataFrame({
            "Dataset split": ["Validation", "Test"],
            "Frozen BERT + Logistic Regression": [
                frozen.get("validation_accuracy", 0),
                frozen_accuracy
            ],
            "Fine-tuned BERT": [
                fine.get("validation_accuracy", 0),
                fine_accuracy
            ]
        }).set_index("Dataset split")

        st.bar_chart(
            accuracy_comparison,
            y_label="Accuracy (0–1)",
            stack=False
        )

        st.dataframe(
            (accuracy_comparison * 100).round(2),
            use_container_width=True
        )

        st.divider()

        # ---------- 3. Validation vs test: F1 ----------
        st.markdown("### 3. F1-score: validation vs test")
        st.caption(
            "Compares each model's F1-score on the validation and test splits."
        )

        f1_comparison = pd.DataFrame({
            "Dataset split": ["Validation", "Test"],
            "Frozen BERT + Logistic Regression": [
                frozen.get("validation_f1", 0),
                frozen_f1
            ],
            "Fine-tuned BERT": [
                fine.get("validation_f1", 0),
                fine_f1
            ]
        }).set_index("Dataset split")

        st.bar_chart(
            f1_comparison,
            y_label="F1-score (0–1)",
            stack=False
        )

        st.dataframe(
            (f1_comparison * 100).round(2),
            use_container_width=True
        )

        st.divider()

        # ---------- 4. Training configuration ----------
        with st.expander("Fine-tuned BERT training configuration"):
            config_view = {
                "Base model": fine.get("model", "Not provided"),
                "Dataset": fine.get("dataset", "Not provided"),
                "Epochs": fine.get("epochs", "Not provided"),
                "Batch size": fine.get("batch_size", "Not provided"),
                "Learning rate": fine.get("learning_rate", "Not provided"),
                "Maximum posts per user": fine.get(
                    "max_posts_per_user", "Not provided"
                ),
                "Maximum tokens per post": fine.get(
                    "max_tokens_per_post", "Not provided"
                )
            }

            st.json(config_view)


# ---------- Live prediction ----------
with prediction_tab:
    st.subheader("Analyse user-provided text")
    st.write(
        "Enter up to 10 posts. Put each post on a separate line. "
        "All entered posts are treated as one user's text history."
    )

    with st.form("prediction_form"):
        user_text = st.text_area(
            "Post text",
            height=220,
            placeholder="Post 1: I have been feeling...\nPost 2: Today was...\nPost 3: ...",
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            nsamples = st.select_slider(
                "SHAP evaluation samples",
                options=[20, 50, 100, 150, 200],
                value=100,
            )
        with col2:
            st.caption("More SHAP samples can improve estimation stability but take longer.")
        submitted = st.form_submit_button(
            "Run BERT + SHAP + Groq",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        posts = [line.strip() for line in user_text.splitlines() if line.strip()]
        if not posts:
            st.error("Please enter at least one non-empty post.")
        elif len(posts) > MAX_POSTS:
            st.warning(
                f"You entered {len(posts)} posts. The model supports up to {MAX_POSTS}; "
                f"only the most recent {MAX_POSTS} entered posts will be used."
            )
            posts = posts[-MAX_POSTS:]

        if posts:
            try:
                with st.spinner("Loading the saved model and calculating SHAP values..."):
                    explainer = get_explainer()
                    result = explainer.explain(posts, nsamples=nsamples)

                st.success("Analysis completed.")
                p1, p2, p3 = st.columns(3)
                p1.metric("Predicted dataset class", result["predicted_class"])
                p2.metric("Positive-class score", f'{result["positive_probability"]:.1%}')
                p3.metric("Posts analysed", str(result["posts_used"]))
                st.caption(
                    f"Model device: {result['device']} · "
                    "The positive-class score is the model's estimated probability for class 1."
                )

                st.markdown("### Positive-class score")
                st.progress(float(result["positive_probability"]))
                st.caption(
                    "This score is not a probability of having depression and is not a clinical risk score."
                )

                st.markdown("### Post-level SHAP contributions")
                shap_df = result["shap_table"][["post_number", "direction", "shap_value", "post"]].copy()
                shap_df["shap_value"] = shap_df["shap_value"].astype(float).round(5)
                st.dataframe(shap_df, use_container_width=True, hide_index=True)

                chart_df = result["shap_table"][["post_number", "shap_value"]].copy()
                chart_df["post_number"] = chart_df["post_number"].astype(str)
                chart_df = chart_df.set_index("post_number")
                st.markdown("#### Contribution chart")
                st.bar_chart(chart_df, y="shap_value", y_label="SHAP value")
                st.caption(
                    "Positive values contribute toward the model's positive class; "
                    "negative values contribute away from it. Values are not direct "
                    "changes in probability and do not imply causation."
                )

                st.markdown("### Groq-generated explanation")
                st.markdown(result["explanation"])

                with st.expander("View exact posts sent to the model"):
                    for index, post in enumerate(posts, start=1):
                        st.markdown(f"**Post {index}:** {post}")

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.markdown(
                    "**Check these items:** model files exist under "
                    "`models/fine_tuned_bert_model/`; `.env` contains a valid "
                    "`GROQ_API_KEY`; and the tokenizer files are present."
                )

# ---------- Methodology ----------
with methodology_tab:
    st.subheader("Text mining and data preparation")
    st.markdown(
        """
        1. Load user-level records from the prepared dataset.
        2. Represent each record as a collection of social-media posts.
        3. Retain up to 10 posts for each user.
        4. Treat the dataset's binary label as the modelling target.
        """
    )
    st.subheader("Natural language processing and classification")
    st.markdown(
        """
        1. Tokenize each post using the BERT tokenizer, with a maximum of 128 tokens.
        2. Generate contextual token representations using BERT.
        3. Mean-pool token representations to form post embeddings.
        4. Aggregate post embeddings into a user representation using the post mask.
        5. Pass the user representation through the classification head to obtain two logits.
        6. Apply softmax to obtain class probabilities.
        """
    )
    st.subheader("Explainability and language-model summary")
    st.markdown(
        """
        - KernelSHAP estimates the contribution of each post relative to the chosen
          zero-post reference baseline.
        - The SHAP table and selected supporting/counter-evidence are sent to Groq.
        - The LLM is instructed to use only the supplied posts and avoid clinical claims.
        - SHAP contributions are model explanations, not causal effects.
        """
    )
    st.warning(
        "Research limitation: the displayed classes are dataset-defined labels. "
        "High test performance does not by itself establish generalization to other "
        "platforms, populations, or real-world mental-health outcomes."
    )
