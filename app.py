import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PunjabiFaith",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,0.08), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(14,165,233,0.08), transparent 30%),
            #f8fafc;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.8rem;
    }

    .research-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 1.4rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 30px rgba(15,23,42,0.06);
        margin-top: 1rem;
    }

    .result-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.25rem;
        color: #111827;
    }

    .review-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        margin-top: 1rem;
    }

    .info-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
    }

    .evidence-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #ffffff;
        border-left: 4px solid #6366f1;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
    }

    .small-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS AND MODELS
# ============================================================

@st.cache_resource
def load_prototype():
    artifacts = joblib.load("punjabifaith_prototype.joblib")
    return artifacts


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def load_nli_model():
    model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model


artifacts = load_prototype()

prototype_model = artifacts["model"]
feature_columns = artifacts["feature_columns"]
label_mapping = artifacts["label_mapping"]

embedding_model = load_embedding_model()
nli_tokenizer, nli_model = load_nli_model()


# ============================================================
# TEXT UTILITIES
# ============================================================

def word_count(text):
    return len(str(text).split())


def extract_numbers(text):
    """
    Extract Arabic and common numeric expressions.
    """
    text = str(text)

    pattern = r"""
        \d+(?:[.,]\d+)*
        |
        \d+/\d+
    """

    return re.findall(pattern, text, flags=re.VERBOSE)


def split_into_sentences(text):
    sentences = re.split(
        r'(?<=[।.!?])\s+',
        str(text).strip()
    )

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 10
    ]


# ============================================================
# EVIDENCE RETRIEVAL
# ============================================================

def retrieve_evidence(article, summary, top_k=3):

    sentences = split_into_sentences(article)

    if len(sentences) == 0:
        return []

    summary_embedding = embedding_model.encode(
        summary,
        convert_to_tensor=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        summary_embedding,
        sentence_embeddings
    )[0]

    k = min(top_k, len(sentences))

    top_indices = torch.topk(
        similarities,
        k=k
    ).indices.cpu().numpy()

    evidence = []

    for idx in top_indices:

        evidence.append({
            "sentence": sentences[idx],
            "similarity": float(similarities[idx])
        })

    return evidence


# ============================================================
# NLI
# ============================================================

def get_nli_scores(premise, hypothesis):

    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = nli_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0].cpu().numpy()

    return {
        "entailment": float(probabilities[0]),
        "neutral": float(probabilities[1]),
        "contradiction": float(probabilities[2])
    }


# ============================================================
# FAITHFULNESS PREDICTION
# ============================================================

def assess_faithfulness(article, generated_summary):

    article = str(article).strip()
    generated_summary = str(generated_summary).strip()

    article_words = word_count(article)
    summary_words = word_count(generated_summary)

    compression_ratio = (
        summary_words / article_words
        if article_words > 0
        else 0
    )

    article_numbers = extract_numbers(article)
    summary_numbers = extract_numbers(generated_summary)

    if len(article_numbers) > 0:

        preserved_numbers = sum(
            1
            for n in summary_numbers
            if n in article_numbers
        )

        number_preservation = (
            preserved_numbers / len(article_numbers)
        )

    else:
        number_preservation = 1.0

    summary_has_number = int(
        len(summary_numbers) > 0
    )

    evidence = retrieve_evidence(
        article,
        generated_summary,
        top_k=3
    )

    entailments = []
    neutrals = []
    contradictions = []
    similarities = []

    for item in evidence:

        scores = get_nli_scores(
            item["sentence"],
            generated_summary
        )

        similarities.append(
            item["similarity"]
        )

        entailments.append(
            scores["entailment"]
        )

        neutrals.append(
            scores["neutral"]
        )

        contradictions.append(
            scores["contradiction"]
        )

    if len(evidence) == 0:

        similarities = [0.0]
        entailments = [0.0]
        neutrals = [1.0]
        contradictions = [0.0]

    features = pd.DataFrame([{

        "article_word_count": article_words,

        "summary_word_count": summary_words,

        "compression_ratio": compression_ratio,

        "article_number_count": len(article_numbers),

        "summary_number_count": len(summary_numbers),

        "number_preservation": number_preservation,

        "summary_has_number": summary_has_number,

        "evidence_similarity_max": max(similarities),

        "evidence_similarity_mean": np.mean(similarities),

        "nli_entailment_max": max(entailments),

        "nli_entailment_mean": np.mean(entailments),

        "nli_neutral_max": max(neutrals),

        "nli_neutral_mean": np.mean(neutrals),

        "nli_contradiction_max": max(contradictions),

        "nli_contradiction_mean": np.mean(contradictions)

    }])

    features = features[feature_columns]

    prediction = prototype_model.predict(
        features
    )[0]

    probabilities = prototype_model.predict_proba(
        features
    )[0]

    predicted_label = label_mapping[prediction]

    model_probability = float(
        max(probabilities)
    )

    if model_probability >= 0.70:

        review_flag = "Lower uncertainty"

    elif model_probability >= 0.50:

        review_flag = "Moderate uncertainty"

    else:

        review_flag = "Human review recommended"

    return {
        "prediction": predicted_label,
        "probabilities": {
            "Faithful": float(probabilities[0]),
            "Partially Faithful": float(probabilities[1]),
            "Not Faithful": float(probabilities[2])
        },
        "model_probability": model_probability,
        "review_flag": review_flag,
        "evidence": evidence,
        "nli": [
            get_nli_scores(
                item["sentence"],
                generated_summary
            )
            for item in evidence
        ],
        "features": features
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="research-badge">MSc DATA SCIENCE • RESEARCH PROTOTYPE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">PunjabiFaith</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Evaluating factual faithfulness in Punjabi abstractive summarization'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(2, gap="large")

with left:

    st.markdown("### 📄 Source Article")

    article = st.text_area(
        "Source article",
        height=330,
        placeholder="Paste the original Punjabi article here...",
        label_visibility="collapsed"
    )

with right:

    st.markdown("### ✍️ Generated Summary")

    summary = st.text_area(
        "Generated summary",
        height=330,
        placeholder="Paste the generated Punjabi summary here...",
        label_visibility="collapsed"
    )


st.markdown("")

assess_button = st.button(
    "🔎  Assess Faithfulness",
    type="primary",
    use_container_width=True
)


# ============================================================
# ASSESSMENT
# ============================================================

if assess_button:

    if not article.strip():

        st.warning(
            "Please provide the source article."
        )

    elif not summary.strip():

        st.warning(
            "Please provide the generated summary."
        )

    else:

        with st.spinner(
            "Analyzing summary faithfulness..."
        ):

            result = assess_faithfulness(
                article,
                summary
            )

        prediction = result["prediction"]

        if prediction == "Faithful":
            icon = "✅"

        elif prediction == "Partially Faithful":
            icon = "🟡"

        else:
            icon = "⚠️"

        st.markdown("---")

        st.markdown("## Assessment Result")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                '<div class="result-card">'
                '<div class="result-label">Predicted faithfulness</div>'
                f'<div class="result-value">{icon} {prediction}</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                '<div class="result-card">'
                '<div class="result-label">Model probability</div>'
                f'<div class="result-value">'
                f'{result["model_probability"] * 100:.1f}%'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                '<div class="result-card">'
                '<div class="result-label">Review status</div>'
                f'<div class="result-value" style="font-size:1.25rem;">'
                f'{result["review_flag"]}'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        if "Human review" in result["review_flag"]:

            st.markdown(
                '<div class="review-card">'
                '⚠️ <b>Human review recommended.</b> '
                'The prototype is uncertain about this prediction.'
                '</div>',
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.markdown("")
        st.markdown("### Probability distribution")

        probabilities = result["probabilities"]

        for label, value in probabilities.items():

            st.write(
                f"**{label}** — {value * 100:.1f}%"
            )

            st.progress(
                value
            )


        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## 🔍 Evidence Explorer")

        st.caption(
            "Top source sentences retrieved using multilingual "
            "semantic similarity, followed by NLI analysis."
        )

        for i, item in enumerate(
            result["evidence"]
        ):

            nli_scores = result["nli"][i]

            st.markdown(
                f'<div class="evidence-card">'
                f'<b>Evidence {i + 1}</b><br><br>'
                f'{item["sentence"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            e1, e2, e3, e4 = st.columns(4)

            with e1:
                st.metric(
                    "Similarity",
                    f'{item["similarity"]:.3f}'
                )

            with e2:
                st.metric(
                    "Entailment",
                    f'{nli_scores["entailment"]:.3f}'
                )

            with e3:
                st.metric(
                    "Neutral",
                    f'{nli_scores["neutral"]:.3f}'
                )

            with e4:
                st.metric(
                    "Contradiction",
                    f'{nli_scores["contradiction"]:.3f}'
                )


        # ----------------------------------------------------
        # AUTOMATIC FEATURES
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## 📊 Assessment Signals")

        feature_values = result["features"].iloc[0]

        f1, f2, f3, f4 = st.columns(4)

        with f1:
            st.metric(
                "Article words",
                int(feature_values["article_word_count"])
            )

        with f2:
            st.metric(
                "Summary words",
                int(feature_values["summary_word_count"])
            )

        with f3:
            st.metric(
                "Compression ratio",
                f'{feature_values["compression_ratio"]:.3f}'
            )

        with f4:
            st.metric(
                "Number preservation",
                f'{feature_values["number_preservation"] * 100:.1f}%'
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'PunjabiFaith • Research prototype for MSc Data Science<br>'
    'This system is a proof-of-concept and not a production factuality verifier.'
    '</div>',
    unsafe_allow_html=True
)
