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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM STYLING — MAROON + WHITE
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background: #f7f3f1;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- GENERAL TEXT ---------- */

    h1, h2, h3 {
        color: #5b1f2a !important;
    }

    p, label, .stMarkdown {
        color: #2f2930;
    }


    /* ---------- HERO ---------- */

    .hero-box {
        background: linear-gradient(
            135deg,
            #5b1f2a 0%,
            #7b2d3b 100%
        );
        padding: 32px 36px;
        border-radius: 20px;
        margin-bottom: 28px;
        box-shadow: 0 8px 25px rgba(91, 31, 42, 0.18);
    }

    .hero-title {
        color: white !important;
        font-size: 2.35rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #f8e9ec !important;
        font-size: 1.08rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .hero-note {
        color: #ead3d8 !important;
        font-size: 0.88rem;
        margin-top: 14px;
        margin-bottom: 0;
    }


    /* ---------- SECTION HEADINGS ---------- */

    .section-heading {
        color: #5b1f2a;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 24px;
        margin-bottom: 10px;
    }


    /* ---------- TEXT AREAS ---------- */

    textarea {
        background-color: white !important;
        color: #252126 !important;
        border: 1.5px solid #d8c5ca !important;
        border-radius: 13px !important;
    }

    textarea:focus {
        border: 1.8px solid #7b2d3b !important;
        box-shadow: 0 0 0 2px rgba(123, 45, 59, 0.10) !important;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        background: #6b2432 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 3rem;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 5px 15px rgba(107, 36, 50, 0.20);
    }

    .stButton > button:hover {
        background: #521b27 !important;
        color: white !important;
        border: none !important;
    }


    /* ---------- RESULT CARDS ---------- */

    .result-card {
        background: white;
        border: 1px solid #e4d8dc;
        border-top: 5px solid #6b2432;
        border-radius: 16px;
        padding: 22px;
        min-height: 130px;
        box-shadow: 0 4px 14px rgba(91, 31, 42, 0.07);
    }

    .result-label {
        color: #7b6670;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .result-value {
        color: #5b1f2a;
        font-size: 1.42rem;
        font-weight: 800;
    }


    /* ---------- NATIVE METRIC ---------- */

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4d8dc;
        border-top: 4px solid #6b2432;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(91, 31, 42, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #78666d !important;
    }

    [data-testid="stMetricValue"] {
        color: #5b1f2a !important;
    }


    /* ---------- PROBABILITY AREA ---------- */

    .probability-card {
        background: white;
        border: 1px solid #e4d8dc;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 14px rgba(91, 31, 42, 0.06);
    }


    /* ---------- EVIDENCE ---------- */

    .evidence-card {
        background: white;
        border-left: 5px solid #7b2d3b;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 3px 12px rgba(91, 31, 42, 0.07);
    }

    .evidence-title {
        color: #6b2432;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 9px;
    }

    .evidence-text {
        color: #252126 !important;
        font-size: 1rem;
        line-height: 1.75;
        margin-bottom: 12px;
    }

    .evidence-meta {
        color: #75666c !important;
        font-size: 0.84rem;
        line-height: 1.7;
    }


    /* ---------- SIGNAL CARDS ---------- */

    .signal-card {
        background: white;
        border: 1px solid #e4d8dc;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(91, 31, 42, 0.06);
    }

    .signal-title {
        color: #6b2432;
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 14px;
    }

    .signal-row {
        display: flex;
        justify-content: space-between;
        padding: 7px 0;
        border-bottom: 1px solid #f0e8eb;
        color: #40363b;
    }

    .signal-value {
        color: #5b1f2a;
        font-weight: 700;
    }


    /* ---------- INFO / WARNING ---------- */

    .info-box {
        background: #f4e8eb;
        border-left: 5px solid #6b2432;
        border-radius: 10px;
        padding: 16px 18px;
        color: #4d2831;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 5px solid #c46a25;
        border-radius: 10px;
        padding: 16px 18px;
        color: #713b17;
    }


    /* ---------- DIVIDER ---------- */

    hr {
        border: none;
        border-top: 1px solid #dfd1d5;
        margin: 30px 0;
    }


    /* ---------- PROGRESS BAR ---------- */

    [data-testid="stProgress"] > div > div > div > div {
        background-color: #6b2432;
    }


    /* ---------- FOOTER ---------- */

    .footer-note {
        color: #786b71;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 30px;
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
# HERO
# ============================================================

st.markdown("""
<div class="hero-box">

    <div class="hero-title">
        🪷 PunjabiFaith
    </div>

    <div class="hero-subtitle">
        Lightweight Faithfulness Assessment for Punjabi
        Abstractive Summarization
    </div>

    <div class="hero-note">
        Evidence retrieval + NLI signals + interpretable
        automatic assessment
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section-heading">① Source Article</div>',
    unsafe_allow_html=True
)

article = st.text_area(
    "Source Article",
    height=260,
    placeholder="Paste the original Punjabi article here...",
    label_visibility="collapsed"
)


st.markdown(
    '<div class="section-heading">② Generated Summary</div>',
    unsafe_allow_html=True
)

summary = st.text_area(
    "Generated Summary",
    height=160,
    placeholder="Paste the generated Punjabi summary here...",
    label_visibility="collapsed"
)


st.markdown("<br>", unsafe_allow_html=True)


assess_button = st.button(
    "🔍  Assess Faithfulness",
    use_container_width=True
)


# ============================================================
# ASSESSMENT
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


    st.divider()


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-heading">③ Assessment Result</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">Predicted Faithfulness</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">{result["prediction"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">Model Confidence</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">{result["confidence"]:.1%}</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Uncalibrated confidence"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">Review Flag</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">{result["risk"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.markdown(
        '<div class="section-heading">④ Prediction Probabilities</div>',
        unsafe_allow_html=True
    )


    probabilities = result["probabilities"]

    probability_df = pd.DataFrame({
        "Faithfulness Class": list(
            probabilities.keys()
        ),
        "Probability": list(
            probabilities.values()
        )
    })

    probability_df = probability_df.set_index(
        "Faithfulness Class"
    )


    # Clean native chart
    st.bar_chart(
        probability_df,
        y="Probability",
        height=280
    )


    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Faithful",
            f'{probabilities["Faithful"]:.1%}'
        )

    with p2:

        st.metric(
            "Partially Faithful",
            f'{probabilities["Partially Faithful"]:.1%}'
        )

    with p3:

        st.metric(
            "Not Faithful",
            f'{probabilities["Not Faithful"]:.1%}'
        )


    # ========================================================
    # REVIEW WARNING
    # ========================================================

    if result["confidence"] < 0.50:

        st.markdown(
            """
            <div class="warning-box">

            <strong>⚠ Human review recommended</strong><br><br>

            The prototype shows relatively high uncertainty
            for this example. The prediction should therefore
            be treated as a review signal rather than a
            definitive factuality decision.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # EVIDENCE EXPLORER
    # ========================================================

    st.markdown(
        '<div class="section-heading">⑤ Evidence Explorer</div>',
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
                <div class="evidence-card">

                    <div class="evidence-title">
                        Evidence {i}
                    </div>

                    <div class="evidence-text">
                        {item["sentence"]}
                    </div>

                    <div class="evidence-meta">
                        Semantic similarity:
                        <strong>{item["similarity"]:.3f}</strong>
                        &nbsp;&nbsp;•&nbsp;&nbsp;

                        NLI entailment:
                        <strong>{item["entailment"]:.3f}</strong>
                        &nbsp;&nbsp;•&nbsp;&nbsp;

                        NLI neutral:
                        <strong>{item["neutral"]:.3f}</strong>
                        &nbsp;&nbsp;•&nbsp;&nbsp;

                        NLI contradiction:
                        <strong>{item["contradiction"]:.3f}</strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # ASSESSMENT SIGNALS
    # ========================================================

    st.markdown(
        '<div class="section-heading">⑥ Assessment Signals</div>',
        unsafe_allow_html=True
    )


    features = result["features"].iloc[0]


    s1, s2 = st.columns(2)


    with s1:

        st.markdown(
            '<div class="signal-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="signal-title">Summary Characteristics</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="signal-row">
                <span>Article words</span>
                <span class="signal-value">
                    {int(features["article_word_count"])}
                </span>
            </div>

            <div class="signal-row">
                <span>Summary words</span>
                <span class="signal-value">
                    {int(features["summary_word_count"])}
                </span>
            </div>

            <div class="signal-row">
                <span>Compression ratio</span>
                <span class="signal-value">
                    {features["compression_ratio"]:.3f}
                </span>
            </div>

            <div class="signal-row">
                <span>Article numbers</span>
                <span class="signal-value">
                    {int(features["article_number_count"])}
                </span>
            </div>

            <div class="signal-row">
                <span>Summary numbers</span>
                <span class="signal-value">
                    {int(features["summary_number_count"])}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with s2:

        st.markdown(
            '<div class="signal-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="signal-title">Evidence & NLI Signals</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="signal-row">
                <span>Max evidence similarity</span>
                <span class="signal-value">
                    {features["evidence_similarity_max"]:.3f}
                </span>
            </div>

            <div class="signal-row">
                <span>Mean evidence similarity</span>
                <span class="signal-value">
                    {features["evidence_similarity_mean"]:.3f}
                </span>
            </div>

            <div class="signal-row">
                <span>Max NLI entailment</span>
                <span class="signal-value">
                    {features["nli_entailment_max"]:.3f}
                </span>
            </div>

            <div class="signal-row">
                <span>Mean NLI entailment</span>
                <span class="signal-value">
                    {features["nli_entailment_mean"]:.3f}
                </span>
            </div>

            <div class="signal-row">
                <span>Max NLI contradiction</span>
                <span class="signal-value">
                    {features["nli_contradiction_max"]:.3f}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-box">

    <strong>Research Prototype</strong><br><br>

    PunjabiFaith is a lightweight proof-of-concept for
    faithfulness assessment of Punjabi abstractive summaries.
    Predictions should be interpreted together with retrieved
    evidence and human review where required.

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer-note">
        PunjabiFaith • MSc Data Science Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)
