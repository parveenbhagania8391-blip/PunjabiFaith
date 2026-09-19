import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.utils.validation import check_is_fitted


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PunjabiFaith",
    page_icon="🪷",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f8fafc 100%
    );
    border: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
}

.hero h1 {
    color: #111827 !important;
    font-size: 2.4rem !important;
    margin-bottom: 0.3rem;
}

.hero p {
    color: #475569 !important;
    font-size: 1.05rem;
}

.section-title {
    color: #111827;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.7rem;
}

.card {
    padding: 1.2rem;
    border-radius: 14px;
    background: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 14px;
    background: white;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.metric-label {
    color: #64748b;
    font-size: 0.9rem;
}

.metric-value {
    color: #111827;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

.evidence-box {
    padding: 1rem;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 0.8rem;
}

.small-text {
    color: #64748b;
    font-size: 0.88rem;
}

.warning-box {
    padding: 1rem;
    border-radius: 12px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
}

.info-box {
    padding: 1rem;
    border-radius: 12px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_prototype():

    artifacts = joblib.load(
        "punjabifaith_prototype.joblib"
    )

    model = artifacts["model"]

    # Confirm that the stored model is fitted
    check_is_fitted(model)

    return artifacts


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def load_nli_model():

    model_name = (
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:

    artifacts = load_prototype()

    prototype_model = artifacts["model"]

    feature_columns = artifacts[
        "feature_columns"
    ]

    label_mapping = artifacts[
        "label_mapping"
    ]

except Exception as e:

    st.error(
        "The prototype model could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# LOAD SUPPORTING MODELS
# ============================================================

try:

    embedding_model = load_embedding_model()

    nli_tokenizer, nli_model = load_nli_model()

except Exception as e:

    st.error(
        "A supporting language model could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# TEXT FUNCTIONS
# ============================================================

def word_count(text):

    return len(
        str(text).split()
    )


def extract_numbers(text):

    text = str(text)

    pattern = r"""
        \d+(?:[.,]\d+)*
        |
        \d+/\d+
    """

    return re.findall(
        pattern,
        text,
        flags=re.VERBOSE
    )


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
# NLI
# ============================================================

def get_nli_scores(
    premise,
    hypothesis
):

    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = nli_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0].cpu().numpy()

    return {

        "entailment":
            float(probabilities[0]),

        "neutral":
            float(probabilities[1]),

        "contradiction":
            float(probabilities[2])
    }


# ============================================================
# EVIDENCE RETRIEVAL
# ============================================================

def retrieve_evidence(
    article,
    summary,
    top_k=3
):

    sentences = split_into_sentences(
        article
    )

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

    k = min(
        top_k,
        len(sentences)
    )

    top_indices = torch.topk(
        similarities,
        k=k
    ).indices.cpu().numpy()

    evidence = []

    for idx in top_indices:

        evidence.append({

            "sentence":
                sentences[idx],

            "similarity":
                float(similarities[idx])
        })

    return evidence


# ============================================================
# MAIN ASSESSMENT FUNCTION
# ============================================================

def assess_faithfulness(
    article,
    generated_summary
):

    # --------------------------------------------------------
    # Surface features
    # --------------------------------------------------------

    article_words = word_count(
        article
    )

    summary_words = word_count(
        generated_summary
    )

    compression_ratio = (
        summary_words / article_words
        if article_words > 0
        else 0
    )

    article_numbers = extract_numbers(
        article
    )

    summary_numbers = extract_numbers(
        generated_summary
    )

    if len(article_numbers) > 0:

        preserved_numbers = sum(
            1
            for n in summary_numbers
            if n in article_numbers
        )

        number_preservation = (
            preserved_numbers /
            len(article_numbers)
        )

    else:

        number_preservation = 1.0

    summary_has_number = int(
        len(summary_numbers) > 0
    )

    surface_features = pd.DataFrame([{

        "article_word_count":
            article_words,

        "summary_word_count":
            summary_words,

        "compression_ratio":
            compression_ratio,

        "article_number_count":
            len(article_numbers),

        "summary_number_count":
            len(summary_numbers),

        "number_preservation":
            number_preservation,

        "summary_has_number":
            summary_has_number
    }])


    # --------------------------------------------------------
    # Evidence retrieval
    # --------------------------------------------------------

    evidence = retrieve_evidence(
        article,
        generated_summary,
        top_k=3
    )

    entailments = []
    neutrals = []
    contradictions = []
    similarities = []

    evidence_results = []

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

        evidence_results.append({

            "sentence":
                item["sentence"],

            "similarity":
                item["similarity"],

            "entailment":
                scores["entailment"],

            "neutral":
                scores["neutral"],

            "contradiction":
                scores["contradiction"]
        })


    # --------------------------------------------------------
    # Empty evidence fallback
    # --------------------------------------------------------

    if len(evidence) == 0:

        similarities = [0.0]

        entailments = [0.0]

        neutrals = [1.0]

        contradictions = [0.0]


    # --------------------------------------------------------
    # NLI features
    # --------------------------------------------------------

    nli_features = pd.DataFrame([{

        "evidence_similarity_max":
            max(similarities),

        "evidence_similarity_mean":
            np.mean(similarities),

        "nli_entailment_max":
            max(entailments),

        "nli_entailment_mean":
            np.mean(entailments),

        "nli_neutral_max":
            max(neutrals),

        "nli_neutral_mean":
            np.mean(neutrals),

        "nli_contradiction_max":
            max(contradictions),

        "nli_contradiction_mean":
            np.mean(contradictions)
    }])


    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    input_features = pd.concat(
        [
            surface_features,
            nli_features
        ],
        axis=1
    )

    input_features = input_features[
        feature_columns
    ]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = prototype_model.predict(
        input_features
    )[0]

    probabilities = (
        prototype_model.predict_proba(
            input_features
        )[0]
    )


    predicted_label = label_mapping[
        prediction
    ]

    confidence = float(
        max(probabilities)
    )


    # --------------------------------------------------------
    # Review flag
    # --------------------------------------------------------

    if confidence >= 0.70:

        risk = "Low uncertainty"

    elif confidence >= 0.50:

        risk = "Moderate uncertainty"

    else:

        risk = (
            "High uncertainty — "
            "human review recommended"
        )


    return {

        "prediction":
            predicted_label,

        "confidence":
            confidence,

        "risk":
            risk,

        "probabilities": {

            "Faithful":
                float(probabilities[0]),

            "Partially Faithful":
                float(probabilities[1]),

            "Not Faithful":
                float(probabilities[2])
        },

        "features":
            input_features,

        "evidence":
            evidence_results
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🪷 PunjabiFaith</h1>

        <p>
        Lightweight faithfulness assessment
        for Punjabi abstractive summaries
        </p>

        <p class="small-text">
        A research proof-of-concept combining
        automatic features, evidence retrieval,
        and NLI-based signals.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">1. Provide the source article</div>',
    unsafe_allow_html=True
)

article = st.text_area(
    "Source Article",
    height=260,
    placeholder=(
        "Paste the original Punjabi article here..."
    ),
    label_visibility="collapsed"
)


st.markdown(
    '<div class="section-title">2. Provide the generated summary</div>',
    unsafe_allow_html=True
)

summary = st.text_area(
    "Generated Summary",
    height=160,
    placeholder=(
        "Paste the generated Punjabi summary here..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# ASSESS BUTTON
# ============================================================

assess_button = st.button(
    "🔍 Assess Faithfulness",
    use_container_width=True
)


# ============================================================
# RUN ASSESSMENT
# ============================================================

if assess_button:

    if not article.strip():

        st.warning(
            "Please enter the source article."
        )

        st.stop()


    if not summary.strip():

        st.warning(
            "Please enter the generated summary."
        )

        st.stop()


    with st.spinner(
        "Analysing summary and retrieving evidence..."
    ):

        result = assess_faithfulness(
            article,
            summary
        )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Assessment Result</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                Predicted Faithfulness
                </div>

                <div class="metric-value">
                {result["prediction"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                Model Confidence (uncalibrated)
                </div>

                <div class="metric-value">
                {result["confidence"]:.1%}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                Review Flag
                </div>

                <div class="metric-value">
                {result["risk"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction probabilities</div>',
        unsafe_allow_html=True
    )

    probabilities = result[
        "probabilities"
    ]

    for label, value in probabilities.items():

        st.write(
            f"**{label}** — {value:.1%}"
        )

        st.progress(
            float(value)
        )


    # ========================================================
    # HUMAN REVIEW WARNING
    # ========================================================

    if result["confidence"] < 0.50:

        st.markdown(
            """
            <div class="warning-box">

            <strong>Human review recommended</strong><br>

            The prototype shows relatively high
            uncertainty for this example. The prediction
            should therefore be treated as a review signal,
            not as a definitive factuality decision.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # EVIDENCE EXPLORER
    # ========================================================

    st.markdown(
        '<div class="section-title">Evidence Explorer</div>',
        unsafe_allow_html=True
    )

    if len(result["evidence"]) == 0:

        st.info(
            "No suitable evidence sentences were retrieved."
        )

    else:

        for i, item in enumerate(
            result["evidence"],
            start=1
        ):

            st.markdown(
                f"""
                <div class="evidence-box">

                <strong>
                Evidence {i}
                </strong>

                <br><br>

                {item["sentence"]}

                <br><br>

                <span class="small-text">
                Semantic similarity:
                {item["similarity"]:.3f}
                &nbsp; | &nbsp;
                NLI entailment:
                {item["entailment"]:.3f}
                &nbsp; | &nbsp;
                NLI neutral:
                {item["neutral"]:.3f}
                &nbsp; | &nbsp;
                NLI contradiction:
                {item["contradiction"]:.3f}
                </span>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # AUTOMATIC SIGNALS
    # ========================================================

    st.markdown(
        '<div class="section-title">Assessment Signals</div>',
        unsafe_allow_html=True
    )

    features = result[
        "features"
    ].iloc[0]


    signal_col1, signal_col2 = st.columns(2)


    with signal_col1:

        st.markdown(
            f"""
            <div class="card">

            <strong>Summary characteristics</strong>

            <br><br>

            Article words:
            <strong>{int(features["article_word_count"])}</strong>

            <br>

            Summary words:
            <strong>{int(features["summary_word_count"])}</strong>

            <br>

            Compression ratio:
            <strong>{features["compression_ratio"]:.3f}</strong>

            <br>

            Article numbers:
            <strong>{int(features["article_number_count"])}</strong>

            <br>

            Summary numbers:
            <strong>{int(features["summary_number_count"])}</strong>

            </div>
            """,
            unsafe_allow_html=True
        )


    with signal_col2:

        st.markdown(
            f"""
            <div class="card">

            <strong>Evidence / NLI signals</strong>

            <br><br>

            Maximum evidence similarity:
            <strong>
            {features["evidence_similarity_max"]:.3f}
            </strong>

            <br>

            Mean evidence similarity:
            <strong>
            {features["evidence_similarity_mean"]:.3f}
            </strong>

            <br>

            Maximum NLI entailment:
            <strong>
            {features["nli_entailment_max"]:.3f}
            </strong>

            <br>

            Mean NLI entailment:
            <strong>
            {features["nli_entailment_mean"]:.3f}
            </strong>

            <br>

            Maximum NLI contradiction:
            <strong>
            {features["nli_contradiction_max"]:.3f}
            </strong>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.markdown(
    """
    <br>

    <div class="info-box">

    <strong>Research prototype</strong><br>

    This system is a proof-of-concept for lightweight
    faithfulness assessment. It is not a production
    factuality verifier. Predictions should be interpreted
    together with the retrieved evidence and, when flagged,
    human review.

    </div>
    """,
    unsafe_allow_html=True
)
