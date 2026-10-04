# PunjabiFaith

A lightweight research prototype for evaluating factual faithfulness in Punjabi abstractive summarization.

## Prototype configuration
- Features: 15 automatic signals (surface, evidence retrieval, NLI)
- Classifier: Logistic Regression
- Data: 300 human-annotated examples
- Validation: 5-fold stratified cross-validation
- Results: 35.7% accuracy, 0.354 Macro-F1 (majority baseline: 36.0%)

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

To enable the AI assessment tab, add `GEMINI_API_KEY` to Streamlit Secrets.

## Limitations
This is a proof-of-concept. Classifier performance is close to the majority-class baseline, so outputs are preliminary research signals and do not replace human factual assessment.
