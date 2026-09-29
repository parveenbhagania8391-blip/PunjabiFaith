import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from sklearn.utils.validation import check_is_fitted
from google import genai
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PunjabiFaith | Faithfulness Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
:root {
    --bg: #0a1020;
    --panel: #151f31;
    --panel-2: #1b273b;
    --border: #26364f;
    --text: #f4f7fb;
    --muted: #91a4bf;
    --teal: #16c6b2;
    --teal-soft: #123f43;
    --maroon: #a23b4a;
    --amber: #f4ad2e;
    --red: #f15b63;
    --blue: #5f8cff;
}
.stApp {
    background: radial-gradient(circle at 78% 5%, rgba(22,198,178,.08), transparent 28%),
                radial-gradient(circle at 20% 10%, rgba(162,59,74,.07), transparent 25%),
                var(--bg);
    color: var(--text);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
section[data-testid="stSidebar"] {
    background: #0d1627;
    border-right: 1px solid var(--border);
    min-width: 340px;
    max-width: 340px;
}
section[data-testid="stSidebar"] * { color: var(--text); }
.sidebar-section {
    margin: 1.15rem 0 .55rem 0;
    color: #6ee7da;
    font-size: .68rem;
    font-weight: 900;
    letter-spacing: .14em;
    text-transform: uppercase;
}
.sidebar-heading {
    margin: 0 0 .65rem 0;
    color: #f4f7fb;
    font-size: 1rem;
    font-weight: 800;
}
.workflow-card {
    background: #111c2e;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .65rem .7rem;
    margin: .35rem 0;
}
.workflow-step {
    display: flex;
    align-items: flex-start;
    gap: .55rem;
    color: #dce6f4;
    font-size: .78rem;
    line-height: 1.35;
}
.workflow-num {
    min-width: 1.35rem;
    height: 1.35rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: rgba(22,198,178,.12);
    border: 1px solid rgba(22,198,178,.28);
    color: #6ee7da !important;
    font-size: .65rem;
    font-weight: 900;
}
.glossary-group {
    margin-top: .85rem;
}
.glossary-group-title {
    color: #91a4bf;
    font-size: .62rem;
    font-weight: 900;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .4rem;
}
.glossary-item {
    background: linear-gradient(145deg, rgba(27,39,59,.82), rgba(17,28,46,.92));
    border: 1px solid #26364f;
    border-left: 3px solid #16c6b2;
    border-radius: 10px;
    padding: .62rem .72rem;
    margin: .42rem 0;
}
.glossary-term {
    color: #f4f7fb;
    font-size: .78rem;
    font-weight: 800;
    margin-bottom: .2rem;
}
.glossary-def {
    color: #91a4bf;
    font-size: .70rem;
    line-height: 1.45;
}
.glossary-note {
    margin-top: .7rem;
    padding: .7rem .75rem;
    background: rgba(244,173,46,.06);
    border: 1px solid rgba(244,173,46,.18);
    border-radius: 10px;
    color: #c8b98f;
    font-size: .68rem;
    line-height: 1.45;
}
.sidebar-config {
    display: grid;
    gap: .38rem;
}
.sidebar-config-item {
    background: #111c2e;
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: .48rem .6rem;
    color: #cbd7e8;
    font-size: .72rem;
}
.sidebar-config-item b { color: #f4f7fb; }
.brand {
    padding: 0.2rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.brand-title { font-size: 1.45rem; font-weight: 800; letter-spacing: .02em; }
.brand-sub { color: var(--teal); font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; margin-top: .2rem; }
.badge {
    display: inline-block;
    border: 1px solid rgba(22,198,178,.45);
    background: rgba(22,198,178,.08);
    color: #6ee7da;
    border-radius: 999px;
    padding: .28rem .7rem;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .11em;
    text-transform: uppercase;
}
.hero-kicker { color: var(--teal); font-size: .75rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.hero-title { font-size: clamp(2.3rem, 5vw, 4.2rem); line-height: .98; font-weight: 800; margin: .35rem 0 .55rem 0; }
.hero-title span { color: var(--teal); }
.hero-sub { color: var(--muted); font-size: 1rem; max-width: 850px; line-height: 1.7; }
.section-label { color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; margin: .4rem 0 .55rem 0; }
.metric-card {
    background: linear-gradient(145deg, rgba(27,39,59,.98), rgba(20,31,48,.98));
    border: 1px solid var(--border);
    border-top: 3px solid var(--teal);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    min-height: 112px;
    box-shadow: 0 12px 30px rgba(0,0,0,.14);
}
.metric-label { color: #91a4bf; font-size: .67rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
.metric-value { color: var(--text); font-size: 1.7rem; font-weight: 800; margin-top: .35rem; }
.metric-note { color: #7186a4; font-size: .72rem; margin-top: .15rem; }
.result-panel {
    background: linear-gradient(145deg, #172338, #121d2f);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 18px 40px rgba(0,0,0,.18);
}
.result-label { color: var(--muted); font-size: .68rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.result-value { font-size: 2.35rem; font-weight: 900; margin-top: .3rem; }
.result-value.good { color: var(--teal); }
.result-value.warn { color: var(--amber); }
.result-value.bad { color: var(--red); }
.callout {
    background: rgba(22,198,178,.06);
    border: 1px solid rgba(22,198,178,.22);
    border-left: 4px solid var(--teal);
    border-radius: 12px;
    padding: .85rem 1rem;
    color: #c9d6e8;
    line-height: 1.55;
}
.review {
    background: rgba(244,173,46,.07);
    border: 1px solid rgba(244,173,46,.25);
    border-left: 4px solid var(--amber);
    border-radius: 12px;
    padding: .8rem 1rem;
    color: #e8d6ad;
}
.evidence-card {
    background: #121d2f;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: .9rem 1rem;
    margin-bottom: .75rem;
}
.evidence-head { color: var(--teal); font-size: .7rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.evidence-text { color: #e6edf7; line-height: 1.7; margin: .35rem 0 .55rem 0; }
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: .18rem .55rem;
    font-size: .66rem;
    font-weight: 800;
    letter-spacing: .05em;
    margin-right: .3rem;
}
.pill-teal { color: #71eadf; background: rgba(22,198,178,.11); border: 1px solid rgba(22,198,178,.2); }
.pill-amber { color: #ffd27a; background: rgba(244,173,46,.10); border: 1px solid rgba(244,173,46,.2); }
.pill-red { color: #ff9da2; background: rgba(241,91,99,.10); border: 1px solid rgba(241,91,99,.2); }
textarea {
    background: #111c2e !important;
    color: #edf3fb !important;
    border-color: var(--border) !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #16c6b2, #12a99a);
    color: #071017;
    border: none;
    border-radius: 11px;
    font-weight: 800;
    letter-spacing: .03em;
    min-height: 2.7rem;
}
div[data-testid="stButton"] > button:hover { filter: brightness(1.08); }
div[data-testid="stTabs"] button {
    color: #8fa3bf;
    font-weight: 700;
}
div[data-testid="stTabs"] button[aria-selected="true"] { color: #16c6b2; }
hr { border-color: var(--border); }
.small-muted { color: var(--muted); font-size: .82rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS / MODELS
# ============================================================
@st.cache_resource
def load_prototype():
    artifacts = joblib.load("punjabifaith_prototype.joblib")
    model = artifacts["model"]
    check_is_fitted(model)
    return artifacts

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_nli_model():
    name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    mdl.eval()
    return tok, mdl

try:
    artifacts = load_prototype()
    prototype_model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    label_mapping = artifacts["label_mapping"]
except Exception as e:
    st.error("Prototype model could not be loaded.")
    st.exception(e)
    st.stop()

embedding_model = load_embedding_model()
nli_tokenizer, nli_model = load_nli_model()

# ============================================================
# HELPERS — SAME RESEARCH BACKEND
# ============================================================
def word_count(text):
    return len(re.findall(r"\S+", str(text)))

def extract_numbers(text):
    return re.findall(r"\d+(?:[.,]\d+)?", str(text))

def split_into_sentences(text):
    sentences = re.split(r'(?<=[।.!?])\s+', str(text).strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

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
    probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    return {
        "entailment": float(probabilities[0]),
        "neutral": float(probabilities[1]),
        "contradiction": float(probabilities[2])
    }

def retrieve_evidence(article, summary, top_k=3):
    sentences = split_into_sentences(article)
    if not sentences:
        return []
    summary_embedding = embedding_model.encode(summary, convert_to_tensor=True)
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    similarities = util.cos_sim(summary_embedding, sentence_embeddings)[0]
    k = min(top_k, len(sentences))
    top_indices = torch.topk(similarities, k=k).indices.cpu().numpy()
    evidence = []
    for idx in top_indices:
        evidence.append({
            "sentence": sentences[idx],
            "similarity": float(similarities[idx])
        })
    return evidence
    
# ============================================================
# AI-ASSISTED FAITHFULNESS JUDGE
# ============================================================

def ai_faithfulness_judge(article, summary, evidence, nli_features):
    evidence_text = "\n".join(
        [
            f"- Evidence: {item['sentence']}\n"
            f"  Similarity: {item['similarity']:.3f}"
            for item in evidence
        ]
    )

    prompt = f"""
You are an evidence-grounded faithfulness evaluator for Punjabi
abstractive summarization.

SOURCE ARTICLE:
{article}

GENERATED SUMMARY:
{summary}

RETRIEVED EVIDENCE:
{evidence_text}

NLI SIGNALS:
{nli_features}

Evaluate the summary ONLY against the source article and retrieved evidence.

Rules:
1. Do not invent evidence.
2. Identify whether the summary is supported, partially supported,
   unsupported, or contradicted by the source.
3. Pay special attention to factual claims, numbers, dates and entities.
4. A semantically similar sentence is NOT automatically factual evidence.
5. If the available evidence is insufficient, say so.
6. Keep the explanation concise.

Return ONLY valid JSON with these fields:

{{
  "overall_assessment": "Faithful | Partially Faithful | Not Faithful",
  "faithfulness_score": 0,
  "supported_claims": [],
  "unsupported_claims": [],
  "contradicted_claims": [],
  "number_date_issue": "Yes | No",
  "entity_issue": "Yes | No",
  "reason": "",
  "human_review_required": "Yes | No"
}}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def analyze_summary(article, generated_summary):
    generated_summary = generated_summary.replace("</s>", "").strip()

def analyze_summary(article, generated_summary):
    generated_summary = generated_summary.replace("</s>", "").strip()

    article_words = word_count(article)
    summary_words = word_count(generated_summary)
    compression_ratio = summary_words / article_words if article_words else 0

    article_numbers = extract_numbers(article)
    summary_numbers = extract_numbers(generated_summary)

    if article_numbers:
        preserved = sum(1 for n in summary_numbers if n in article_numbers)
        number_preservation = preserved / len(article_numbers)
    else:
        number_preservation = 1.0

    number_status = "available" if article_numbers else "not_applicable"

    surface = pd.DataFrame([{
        "article_word_count": article_words,
        "summary_word_count": summary_words,
        "compression_ratio": compression_ratio,
        "article_number_count": len(article_numbers),
        "summary_number_count": len(summary_numbers),
        "number_preservation": number_preservation,
        "summary_has_number": int(bool(summary_numbers)),
    }])

    evidence = retrieve_evidence(article, generated_summary, top_k=3)

    similarities, entailments, neutrals, contradictions = [], [], [], []
    for item in evidence:
        scores = get_nli_scores(item["sentence"], generated_summary)
        similarities.append(item["similarity"])
        entailments.append(scores["entailment"])
        neutrals.append(scores["neutral"])
        contradictions.append(scores["contradiction"])
        item["nli"] = scores

    if not evidence:
        similarities = [0.0]
        entailments = [0.0]
        neutrals = [1.0]
        contradictions = [0.0]

    nli_features = pd.DataFrame([{
        "evidence_similarity_max": max(similarities),
        "evidence_similarity_mean": np.mean(similarities),
        "nli_entailment_max": max(entailments),
        "nli_entailment_mean": np.mean(entailments),
        "nli_neutral_max": max(neutrals),
        "nli_neutral_mean": np.mean(neutrals),
        "nli_contradiction_max": max(contradictions),
        "nli_contradiction_mean": np.mean(contradictions),
    }])

    input_features = pd.concat([surface, nli_features], axis=1)
    input_features = input_features[feature_columns]

    prediction = int(prototype_model.predict(input_features)[0])
    probabilities = prototype_model.predict_proba(input_features)[0]

    labels = {int(k): v for k, v in label_mapping.items()}
    predicted_label = labels[prediction]

    return {
        "prediction": predicted_label,
        "probabilities": {
            "Faithful": float(probabilities[0]),
            "Partially Faithful": float(probabilities[1]),
            "Not Faithful": float(probabilities[2]),
        },
        "features": input_features.iloc[0].to_dict(),
        "evidence": evidence,
        "surface": surface.iloc[0].to_dict(),
        "number_status": number_status,
        "summary": generated_summary,
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-title">🧬 PunjabiFaith</div>
        <div class="brand-sub">Faithfulness Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="badge">Research Prototype</span>', unsafe_allow_html=True)

    # --- Workflow ---
    st.markdown('<div class="sidebar-section">Workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">How the assessment works</div>', unsafe_allow_html=True)
    workflow = [
        "Generated summary",
        "Automatic signals",
        "Evidence retrieval",
        "NLI analysis",
        "Prototype classifier",
        "Human review when needed",
    ]
    for i, step in enumerate(workflow, start=1):
        st.markdown(
            f'<div class="workflow-card"><div class="workflow-step">'
            f'<span class="workflow-num">{i}</span><span>{step}</span></div></div>',
            unsafe_allow_html=True
        )

    st.divider()

    # --- Glossary ---
    st.markdown('<div class="sidebar-section">Research glossary</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">Key terms, in simple language</div>', unsafe_allow_html=True)

    glossary_groups = [
        ("Core concepts", [
            ("Faithfulness", "Whether the generated summary is factually supported by the source article."),
            ("Hallucination", "Information introduced in the summary that is not supported by the source."),
            ("Evidence", "Source-article sentence retrieved as a potential support for the summary."),
        ]),
        ("Evidence & retrieval", [
            ("Semantic similarity", "Measures how closely the summary and a retrieved source sentence are related in meaning."),
        ]),
        ("NLI", [
            ("Natural Language Inference (NLI)", "Estimates whether the retrieved evidence supports, does not clearly support, or contradicts the summary."),
            ("Entailment", "The retrieved evidence supports the generated summary claim."),
            ("Neutral", "The evidence is related but does not clearly support or contradict the claim."),
            ("Contradiction", "The evidence conflicts with the generated summary claim."),
        ]),
        ("Automatic signals", [
            ("Number preservation", "Checks whether numerical information from the source is retained in the summary."),
            ("Compression ratio", "Generated summary length divided by source article length."),
        ]),
    ]

    for group_name, terms in glossary_groups:
        st.markdown(f'<div class="glossary-group"><div class="glossary-group-title">{group_name}</div></div>', unsafe_allow_html=True)
        for term, definition in terms:
            st.markdown(
                f'<div class="glossary-item"><div class="glossary-term">{term}</div>'
                f'<div class="glossary-def">{definition}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="glossary-note"><b>Important:</b> Similarity is a retrieval signal, not proof of factual support. '
        'NLI outputs should be interpreted together with the retrieved evidence and human review.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # --- Configuration ---
    st.markdown('<div class="sidebar-section">Prototype configuration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-config">'
        '<div class="sidebar-config-item"><b>Features</b> · 15 automatic signals</div>'
        '<div class="sidebar-config-item"><b>Classifier</b> · Logistic Regression</div>'
        '<div class="sidebar-config-item"><b>Data</b> · 100 annotated examples</div>'
        '<div class="sidebar-config-item"><b>Validation</b> · 5-fold stratified CV</div>'
        '<div class="sidebar-config-item"><b>CV performance</b> · 33.0% accuracy · 0.302 Macro-F1</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.write("")
    st.caption("Proof-of-concept outputs are preliminary and should not replace human factual assessment.")

# ============================================================

# HERO
# ============================================================
st.markdown('<div class="hero-kicker">Punjabi abstractive summarization</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Evidence-Grounded<br><span>Faithfulness Intelligence</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A lightweight research prototype for examining factual faithfulness in Punjabi generated summaries using surface signals, retrieved evidence, and natural language inference.</div>',
    unsafe_allow_html=True
)
st.write("")

# ============================================================
# INPUT
# ============================================================
st.markdown('<div class="section-label">01 / Provide research input</div>', unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    article = st.text_area(
        "Source article",
        height=300,
        placeholder="Paste the original Punjabi source article here..."
    )

with right:
    generated_summary = st.text_area(
        "Generated summary",
        height=300,
        placeholder="Paste the generated Punjabi summary here..."
    )

run = st.button("▶  ANALYZE SUMMARY", use_container_width=True)

if run:
    if not article.strip() or not generated_summary.strip():
        st.warning("Please provide both the source article and generated summary.")
        st.stop()

    with st.spinner("Running evidence retrieval and NLI analysis..."):
        result = analyze_summary(article, generated_summary)

    # ========================================================
    # RESULTS HEADER
    # ========================================================
    st.divider()
    st.markdown('<div class="section-label">02 / Assessment overview</div>', unsafe_allow_html=True)

    pred = result["prediction"]
    if pred == "Faithful":
        result_class = "good"
        result_note = "The prototype classified the summary as faithful."
    elif pred == "Partially Faithful":
        result_class = "warn"
        result_note = "The prototype classified the summary as partially faithful."
    else:
        result_class = "bad"
        result_note = "The prototype flagged the summary for factual review."

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown(
            f'<div class="result-panel"><div class="result-label">Predicted faithfulness category</div>'
            f'<div class="result-value {result_class}">{pred}</div>'
            f'<div class="small-muted">{result_note}</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        probs = result["probabilities"]
        top_score = max(probs.values())
        st.markdown(
            f'<div class="result-panel"><div class="result-label">Automated model score</div>'
            f'<div class="result-value">{top_score:.2f}</div>'
            f'<div class="small-muted">Uncalibrated classifier probability — not a validated confidence measure.</div></div>',
            unsafe_allow_html=True
        )

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    surface = result["surface"]

    number_metric_value = (
        f'{surface["number_preservation"]*100:.1f}%'
        if result["number_status"] == "available"
        else "N/A"
    )
    number_metric_note = (
        "Source numbers preserved"
        if result["number_status"] == "available"
        else "No source numbers detected"
    )

    metrics = [
        ("SUMMARY", f'{surface["summary_word_count"]:.0f} words', "Generated summary"),
        ("SOURCE", f'{surface["article_word_count"]:.0f} words', "Original article"),
        ("COMPRESSION", f'{surface["compression_ratio"]*100:.2f}%', "Summary / source length"),
        ("NUMBER PRESERVATION", number_metric_value, number_metric_note),
    ]
    for col, (label, value, note) in zip([m1,m2,m3,m4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    tabs = st.tabs(["📊 OVERVIEW", "🔎 EVIDENCE", "🧠 NLI ANALYSIS", "📐 SIGNALS", "📄 RESEARCH VIEW"])

    # ========================================================
    # OVERVIEW
    # ========================================================
    with tabs[0]:
        st.markdown("### Why this result?")
        reasons = []
        if result["number_status"] == "available" and surface["number_preservation"] < 0.5:
            reasons.append("Low numerical preservation was detected.")
        elif result["number_status"] == "not_applicable":
            reasons.append("No source numbers were detected, so numerical preservation is not applicable for this input.")

        if result["features"]["nli_entailment_max"] >= 0.7:
            reasons.append("At least one retrieved evidence sentence strongly supports the generated summary under NLI.")
        elif result["features"]["nli_entailment_max"] < 0.3:
            reasons.append("The strongest retrieved evidence shows limited NLI entailment.")

        if result["features"]["nli_contradiction_max"] >= 0.5:
            reasons.append("A retrieved evidence candidate produced a strong contradiction signal.")
        elif result["features"]["nli_contradiction_max"] >= 0.3:
            reasons.append("A moderate contradiction signal was detected in the retrieved evidence.")

        if surface["compression_ratio"] < 0.02:
            reasons.append("The generated summary is highly compressed relative to the source article.")

        if not reasons:
            reasons.append("The prototype combines multiple automatic signals; the category should be interpreted with the evidence and NLI analysis below.")

        for reason in reasons:
            st.markdown(f"- {reason}")

        st.markdown(
            '<div class="signal-box"><div class="signal-title">Interpretation rule</div>'
            '<div class="signal-line">The predicted category is a classifier output, not a ground-truth factual verdict. '
            'Evidence similarity indicates retrieval relevance; NLI indicates a support/neutral/contradiction signal; '
            'neither alone establishes factual correctness.</div></div>',
            unsafe_allow_html=True
        )

        st.write("")
        st.markdown("### Research interpretation")
        st.markdown(
            '<div class="callout"><b>Important:</b> The displayed category is a preliminary automated assessment. '
            'The prototype was developed as a proof-of-concept and should be interpreted together with the retrieved evidence and NLI signals.</div>',
            unsafe_allow_html=True
        )

        st.write("")
        st.markdown("### 👤 Human review")
        st.markdown(
            '<div class="review"><b>Human verification recommended.</b><br>'
            'Automated faithfulness signals can identify potentially problematic summaries, but they do not replace detailed human factual assessment.</div>',
            unsafe_allow_html=True
        )

    # ========================================================
    # EVIDENCE
    # ========================================================
    with tabs[1]:
        st.markdown("### Retrieved source evidence")
        st.caption("Top-3 source sentences retrieved using multilingual semantic similarity. Similarity measures retrieval relevance; it is not itself proof of factual support.")

        if result["evidence"]:
            for i, item in enumerate(result["evidence"], start=1):
                nli = item["nli"]
                dominant = max(nli, key=nli.get)
                cls = "pill-teal" if dominant == "entailment" else ("pill-red" if dominant == "contradiction" else "pill-amber")
                st.markdown(
                    f'<div class="evidence-card"><div class="evidence-head">Evidence {i}</div>'
                    f'<div class="evidence-text">{item["sentence"]}</div>'
                    f'<span class="pill pill-teal">Similarity {item["similarity"]:.3f}</span>'
                    f'<span class="pill {cls}">NLI {dominant.title()}</span>'
                    f'<span class="pill pill-amber">Entail {nli["entailment"]:.3f}</span>'
                    f'<span class="pill pill-red">Contradict {nli["contradiction"]:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No suitable evidence sentence was retrieved.")

    # ========================================================
    # NLI
    # ========================================================
    with tabs[2]:
        st.markdown("### Evidence-grounded NLI distribution")
        st.caption("These scores describe the relationship between each retrieved evidence candidate and the generated summary.")

        nli_max = {
            "Entailment": result["features"]["nli_entailment_max"],
            "Neutral": result["features"]["nli_neutral_max"],
            "Contradiction": result["features"]["nli_contradiction_max"],
        }
        nli_classes = {
            "Entailment": ("The retrieved evidence supports the generated summary claim.", "#16c6b2"),
            "Neutral": ("The evidence is related but does not clearly support or contradict the claim.", "#f4ad2e"),
            "Contradiction": ("The evidence conflicts with the generated summary claim.", "#f15b63"),
        }
        for name, value in nli_max.items():
            desc, color = nli_classes[name]
            st.markdown(
                f'<div class="nli-row"><div class="nli-head"><span>{name}</span><span>{value:.3f}</span></div>'
                f'<div class="nli-track"><div class="nli-fill" style="width:{value*100:.1f}%;background:{color};"></div></div>'
                f'<div class="small-muted">{desc}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("### Why NLI matters")
        st.markdown(
            '<div class="callout"><b>NLI = Natural Language Inference.</b> In this prototype, it is used after evidence retrieval to estimate whether a retrieved source sentence supports, '
            'does not directly support, or contradicts the generated summary. High semantic similarity alone is not treated as factual support.</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Important limitation")
        st.caption("The current prototype evaluates the whole generated summary against the top retrieved evidence sentences rather than performing fully claim-by-claim verification. Therefore, NLI scores are research signals, not definitive factuality judgments.")

    # ========================================================
    # SIGNALS
    # ========================================================
    with tabs[3]:
        st.markdown("### Automatic assessment signals")
        st.caption("These are the same automatic features used by the prototype classifier. Human annotation fields are not used as model inputs.")
        sig = result["features"]

        signal_rows = [
            ("Article word count", f'{sig["article_word_count"]:.0f}'),
            ("Summary word count", f'{sig["summary_word_count"]:.0f}'),
            ("Compression ratio", f'{sig["compression_ratio"]*100:.2f}%'),
            ("Article number count", f'{sig["article_number_count"]:.0f}'),
            ("Summary number count", f'{sig["summary_number_count"]:.0f}'),
            ("Number preservation", f'{sig["number_preservation"]*100:.1f}%'),
            ("Maximum evidence similarity", f'{sig["evidence_similarity_max"]:.3f}'),
            ("Mean evidence similarity", f'{sig["evidence_similarity_mean"]:.3f}'),
            ("Maximum NLI entailment", f'{sig["nli_entailment_max"]:.3f}'),
            ("Mean NLI entailment", f'{sig["nli_entailment_mean"]:.3f}'),
            ("Maximum NLI neutral", f'{sig["nli_neutral_max"]:.3f}'),
            ("Maximum NLI contradiction", f'{sig["nli_contradiction_max"]:.3f}'),
        ]
        df_sig = pd.DataFrame(signal_rows, columns=["Signal", "Value"])
        st.dataframe(df_sig, use_container_width=True, hide_index=True)

    # ========================================================
    # RESEARCH VIEW
    # ========================================================
    with tabs[4]:
        st.markdown("### Prototype configuration")
        a,b,c,d,e = st.columns(5)
        for col, title, value in [
            (a, "ANNOTATED EXAMPLES", "100"),
            (b, "AUTOMATIC FEATURES", "15"),
            (c, "CLASSIFIER", "Logistic Regression"),
            (d, "CV ACCURACY", "33.0%"),
            (e, "MACRO-F1", "0.302"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{title}</div>'
                    f'<div class="metric-value" style="font-size:1.25rem">{value}</div></div>',
                    unsafe_allow_html=True
                )
        st.write("")
        st.markdown("### Research note")
        st.markdown(
            '<div class="callout"><b>Proof-of-concept:</b> The prototype explores whether automatically derived '
            'surface and evidence/NLI signals can support lightweight faithfulness assessment. '
            'The reported 5-fold CV accuracy (33.0%) and macro-F1 (0.302) indicate that the current feature/classifier '
            'configuration has limited predictive strength on this small annotated sample. It is not presented as a production-grade or fully generalized detector.</div>',
            unsafe_allow_html=True
        )
