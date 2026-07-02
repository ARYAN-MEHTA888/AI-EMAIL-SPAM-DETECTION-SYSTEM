import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)


st.set_page_config(
    page_title="SpamGuard AI — Spam Email Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)



INK = "#0A0E16"
PANEL = "#121826"
PANEL_2 = "#0E1420"
LINE = "#212B3B"
TEXT = "#E8EEF5"
MUTED = "#7C8AA0"
SIGNAL = "#2DD4BF"    
THREAT = "#FB4570"     
AMBER = "#F5B642"
BLUE = "#4D8DFF"
VIOLET = "#7C6CF6"

FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
"""

CUSTOM_CSS = f"""
<style>
{FONT_CSS}

:root {{
    --ink: {INK}; --panel: {PANEL}; --panel-2: {PANEL_2}; --line: {LINE};
    --text: {TEXT}; --muted: {MUTED}; --signal: {SIGNAL}; --threat: {THREAT};
    --amber: {AMBER}; --blue: {BLUE}; --violet: {VIOLET};
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
    background: radial-gradient(120% 140% at 10% -10%, #11182A 0%, var(--ink) 45%) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}}

[data-testid="stHeader"] {{ background: transparent !important; }}

[data-testid="stSidebar"] {{
    background: var(--panel) !important;
    border-right: 1px solid var(--line);
}}
[data-testid="stSidebar"] * {{ color: var(--text) !important; }}

.block-container {{ padding-top: 1.1rem; max-width: 1220px; }}

h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }}
p, span, label, div {{ font-family: 'Inter', sans-serif; }}

/* ---------- Sidebar brand ---------- */

.brand-wrap {{
    display: flex; flex-direction: column; align-items: center; text-align: center;
    padding: 6px 0 16px 0;
}}
.brand-icon {{
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, var(--violet), var(--blue));
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; margin-bottom: 10px;
}}
.brand-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 18px; color: var(--text); }}
.brand-title .accent {{ background: linear-gradient(135deg, var(--violet), var(--blue));
    -webkit-background-clip: text; background-clip: text; color: transparent; }}
.brand-sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

/* ---------- Sidebar nav buttons ---------- */

[data-testid="stSidebar"] .stButton>button {{
    width: 100%; text-align: left !important; justify-content: flex-start !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    font-size: 14px !important; padding: 0.55rem 0.9rem !important;
    border-radius: 10px !important; margin-bottom: 2px;
}}
[data-testid="stSidebar"] button[kind="primary"] {{
    background: linear-gradient(135deg, var(--violet), var(--blue)) !important;
    color: white !important; border: none !important;
    box-shadow: 0 4px 14px rgba(124,108,246,0.35);
}}
[data-testid="stSidebar"] button[kind="secondary"] {{
    background: transparent !important; color: var(--muted) !important;
    border: 1px solid transparent !important; box-shadow: none !important;
}}
[data-testid="stSidebar"] button[kind="secondary"]:hover {{
    background: var(--panel-2) !important; color: var(--text) !important;
}}

.sidebar-shield {{
    margin-top: 18px; text-align: center; padding: 22px 10px;
    border-radius: 14px; background: var(--panel-2); border: 1px solid var(--line);
}}
.sidebar-shield .icon {{ font-size: 40px; filter: drop-shadow(0 0 14px rgba(124,108,246,0.55)); }}
.sidebar-badge {{
    display: inline-flex; align-items: center; gap: 6px; margin-top: 10px;
    background: rgba(124,108,246,0.14); border: 1px solid rgba(124,108,246,0.35);
    color: var(--violet) !important; font-size: 11.5px; padding: 4px 10px; border-radius: 20px;
}}

/* ---------- Top bar ---------- */

.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; margin-bottom: 18px;
}}
.topbar-search {{
    flex: 1; display: flex; align-items: center; gap: 8px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 9px 14px; color: var(--muted); font-size: 13.5px; max-width: 420px;
}}
.topbar-search .kbd {{
    margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px; padding: 1px 6px;
}}
.topbar-right {{ display: flex; align-items: center; gap: 14px; }}
.topbar-bell {{
    position: relative; width: 36px; height: 36px; border-radius: 10px;
    background: var(--panel); border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}}
.topbar-bell .dot {{
    position: absolute; top: 6px; right: 7px; width: 7px; height: 7px;
    border-radius: 50%; background: var(--threat); border: 1.5px solid var(--panel);
}}
.topbar-avatar {{ display: flex; align-items: center; gap: 8px; }}
.topbar-avatar .av {{
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, var(--violet), var(--blue));
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 12.5px; color: white;
}}
.topbar-avatar .meta .name {{ font-size: 13px; font-weight: 600; color: var(--text); line-height: 1.2; }}
.topbar-avatar .meta .role {{ font-size: 11px; color: var(--muted); line-height: 1.2; }}

/* ---------- KPI cards ---------- */

.kpi-card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 16px 18px; display: flex; flex-direction: column; gap: 8px; height: 100%;
}}
.kpi-top {{ display: flex; align-items: center; justify-content: space-between; }}
.kpi-label {{ color: var(--muted); font-size: 12.5px; }}
.kpi-icon {{
    width: 34px; height: 34px; border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 16px; flex-shrink: 0;
}}
.kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 600; color: var(--text); }}
.kpi-caption {{ color: var(--muted); font-size: 11.5px; }}
.kpi-badge {{
    display: inline-flex; font-size: 10.5px; padding: 2px 8px; border-radius: 10px;
    background: rgba(45,212,191,0.16); color: var(--signal); border: 1px solid rgba(45,212,191,0.35);
    width: fit-content;
}}

/* ---------- Generic panel ---------- */

.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 22px 24px; margin-bottom: 18px; }}
.panel-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 600; color: var(--text);
    margin-bottom: 4px; display: flex; align-items: center; gap: 8px; justify-content: space-between; }}
.panel-caption {{ color: var(--muted); font-size: 12.5px; margin-bottom: 16px; }}

.pill {{ font-size: 11px; padding: 3px 10px; border-radius: 20px; border: 1px solid var(--line); }}
.pill.high {{ color: var(--threat); border-color: rgba(251,69,112,0.4); background: rgba(251,69,112,0.1); }}
.pill.low {{ color: var(--signal); border-color: rgba(45,212,191,0.4); background: rgba(45,212,191,0.1); }}

/* ---------- Verdict ---------- */

.verdict {{ border-radius: 16px; padding: 22px; display: flex; align-items: center; gap: 20px;
    border: 1px solid var(--line); margin-bottom: 14px; }}
.verdict.spam {{ background: linear-gradient(135deg, rgba(251,69,112,0.14), rgba(251,69,112,0.03)); border-color: rgba(251,69,112,0.35); }}
.verdict.ham {{ background: linear-gradient(135deg, rgba(45,212,191,0.14), rgba(45,212,191,0.03)); border-color: rgba(45,212,191,0.35); }}
.verdict-ring {{
    width: 64px; height: 64px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 28px;
}}
.verdict.spam .verdict-ring {{ background: rgba(251,69,112,0.16); border: 2px solid var(--threat); }}
.verdict.ham .verdict-ring {{ background: rgba(45,212,191,0.16); border: 2px solid var(--signal); }}
.verdict-label {{ font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; }}
.verdict.spam .verdict-label {{ color: var(--threat); }}
.verdict.ham .verdict-label {{ color: var(--signal); }}
.verdict-detail {{ color: var(--muted); font-size: 12.5px; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }}
.verdict-bar-track {{ height: 7px; border-radius: 6px; background: var(--line); margin-top: 12px; overflow: hidden; }}
.verdict-bar-fill {{ height: 100%; border-radius: 6px; }}

/* ---------- Reason tags ---------- */

.tag {{
    display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500;
    padding: 6px 12px; border-radius: 8px; margin: 3px 6px 3px 0; border: 1px solid transparent;
}}
.tag.amber {{ background: rgba(245,182,66,0.14); color: var(--amber); border-color: rgba(245,182,66,0.35); }}
.tag.red {{ background: rgba(251,69,112,0.14); color: var(--threat); border-color: rgba(251,69,112,0.35); }}
.tag.green {{ background: rgba(45,212,191,0.14); color: var(--signal); border-color: rgba(45,212,191,0.35); }}
.tag.violet {{ background: rgba(124,108,246,0.14); color: var(--violet); border-color: rgba(124,108,246,0.35); }}
.tag.blue {{ background: rgba(77,141,255,0.14); color: var(--blue); border-color: rgba(77,141,255,0.35); }}

/* ---------- Donut ---------- */

.donut-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 14px; }}
.donut {{
    width: 150px; height: 150px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}}
.donut-inner {{
    width: 110px; height: 110px; border-radius: 50%; background: var(--panel);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}}
.donut-inner .pct {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 22px; color: var(--text); }}
.donut-inner .lab {{ font-size: 11px; color: var(--muted); }}
.donut-legend {{ display: flex; gap: 18px; font-size: 12px; color: var(--muted); }}
.donut-legend .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }}

/* ---------- Email details list ---------- */

.detail-row {{ display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid var(--line); font-size: 13px; }}
.detail-row:last-child {{ border-bottom: none; }}
.detail-row .k {{ color: var(--muted); }}
.detail-row .v {{ color: var(--text); font-family: 'JetBrains Mono', monospace; font-weight: 500; }}

/* ---------- Bar list (model performance / top tokens) ---------- */

.bar-row {{ display: flex; align-items: center; gap: 12px; padding: 9px 0; }}
.bar-row .bicon {{
    width: 28px; height: 28px; border-radius: 8px; background: var(--panel-2); border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center; font-size: 13px; flex-shrink: 0;
}}
.bar-row .blabel {{ width: 150px; font-size: 13px; color: var(--text); flex-shrink: 0; }}
.bar-row .btrack {{ flex: 1; height: 8px; border-radius: 6px; background: var(--line); overflow: hidden; }}
.bar-row .bfill {{ height: 100%; border-radius: 6px; }}
.bar-row .bval {{ width: 56px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--text); flex-shrink: 0; }}

/* ---------- Confusion matrix ---------- */

.cm-grid {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; font-size: 12.5px; max-width: 320px; }}
.cm-cell {{ border-radius: 10px; padding: 16px 8px; text-align: center; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 17px; }}
.cm-head {{ color: var(--muted); display: flex; align-items: center; justify-content: center; font-family: 'Inter', sans-serif; font-weight: 500; }}
.cm-rowlabel {{ color: var(--muted); display: flex; align-items: center; font-family: 'Inter', sans-serif; font-weight: 500; writing-mode: vertical-rl; text-orientation: mixed; justify-content: center; }}

/* ---------- Pipeline diagram ---------- */

.pipeline {{ display: flex; align-items: stretch; gap: 0; overflow-x: auto; padding: 6px 0 4px 0; }}
.pipe-step {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; min-width: 160px; flex-shrink: 0; }}
.pipe-step .pn {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--violet); letter-spacing: 0.08em; }}
.pipe-step .pt {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14px; margin-top: 4px; color: var(--text); }}
.pipe-step .pd {{ font-size: 11.5px; color: var(--muted); margin-top: 4px; }}
.pipe-arrow {{ display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 20px; padding: 0 10px; flex-shrink: 0; }}

/* ---------- Widgets ---------- */

.stTextArea textarea {{ background: var(--panel-2) !important; color: var(--text) !important; border: 1px solid var(--line) !important;
    border-radius: 12px !important; font-size: 14.5px !important; }}

.main .stButton>button {{
    background: linear-gradient(135deg, var(--violet), var(--blue)) !important; color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important; font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.55rem 1.2rem !important; transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.main .stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 18px rgba(124,108,246,0.35); }}
.main button[kind="secondary"] {{ background: var(--panel-2) !important; border: 1px solid var(--line) !important; color: var(--text) !important; box-shadow: none !important; }}

[data-testid="stMetric"] {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; }}
[data-testid="stMetricValue"] {{ color: var(--text); font-family: 'JetBrains Mono', monospace; }}
[data-testid="stMetricLabel"] {{ color: var(--muted); }}

[data-testid="stFileUploader"] {{ background: var(--panel-2); border: 1px dashed var(--line); border-radius: 12px; padding: 6px; }}

hr {{ border-color: var(--line) !important; }}

.footer-line {{ text-align: center; color: var(--muted); font-size: 12.5px; padding: 18px 0 6px 0; font-family: 'JetBrains Mono', monospace; }}
.footer-line b {{ color: var(--text); }}
a {{ color: var(--signal) !important; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")


@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf.pkl")


model = load_model()
vectorizer = load_vectorizer()

MODEL_NAME = type(model).__name__
VOCAB_SIZE = len(vectorizer.vocabulary_)
HAS_PROBA = hasattr(model, "predict_proba")
HAS_DECISION = hasattr(model, "decision_function")
HAS_COEF = hasattr(model, "coef_")


TRAIN_FACTS = {
    "dataset_size": 5572,
    "train_size": 4457,
    "test_size": 1115,
    "candidates": ["Naive Bayes", "Logistic Regression", "Random Forest", "Linear SVM"],
    "vectorizer": "TfidfVectorizer(stop_words='english')",
}



URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
SHORTLINK_RE = re.compile(r"(bit\.ly|tinyurl|t\.co|goo\.gl|is\.gd|ow\.ly)\S*", re.I)
SPACE_RE = re.compile(r"\s+")
URGENCY_RE = re.compile(r"\b(now|urgent|hurry|act fast|limited time|expires?|today only|immediately|asap|don'?t miss|last chance)\b", re.I)
PROMO_RE = re.compile(r"\b(free|win|winner|prize|discount|offer|deal|reward|bonus|cash|claim|congratulations)\b", re.I)
MONEY_RE = re.compile(r"(£|\$|€)\s?\d|\b\d+\s?(p|pounds|dollars|usd)\b", re.I)
PHONE_RE = re.compile(r"\b\d{4,}\b")


def normalize_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def score_message(message: str):
    clean = normalize_text(message)
    vec = vectorizer.transform([clean])
    label = int(model.predict(vec)[0])

    if HAS_PROBA:
        proba = model.predict_proba(vec)[0]
        confidence = float(proba.max() * 100)
        raw_score = float(proba[1] - proba[0])
    elif HAS_DECISION:
        raw_score = float(model.decision_function(vec)[0])
        squashed = 1 / (1 + np.exp(-raw_score))
        confidence = float((squashed if label == 1 else 1 - squashed) * 100)
    else:
        confidence, raw_score = 50.0, None

    return label, confidence, raw_score, vec


def message_top_tokens(vec, n=6):
    """Per-message explainability via coef_ * tfidf weight."""
    if not HAS_COEF:
        return [], []
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    arr = vec.toarray()[0]
    contrib = arr * coefs
    nz = np.nonzero(arr)[0]
    if len(nz) == 0:
        return [], []
    nz_sorted = nz[np.argsort(contrib[nz])]
    spam_idx = [i for i in nz_sorted[::-1] if contrib[i] > 0][:n]
    ham_idx = [i for i in nz_sorted if contrib[i] < 0][:n]
    return ([(feature_names[i], contrib[i]) for i in spam_idx],
             [(feature_names[i], contrib[i]) for i in ham_idx])


@st.cache_data
def global_top_tokens(n=8):
    """Model-wide explainability: the strongest spam-indicative tokens
    the Linear SVM actually learned (sorted by raw coefficient)."""
    if not HAS_COEF:
        return []
    names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_idx = np.argsort(coefs)[::-1][:n]
    return [(names[i], float(coefs[i])) for i in top_idx]


def reason_tags(message: str, label: int):
    tags = []
    if SHORTLINK_RE.search(message):
        tags.append(("Shortened Link", "red"))
    elif URL_RE.search(message):
        tags.append(("Contains Link", "blue"))
    if PROMO_RE.search(message):
        tags.append(("Promotional Content", "green"))
    if URGENCY_RE.search(message):
        tags.append(("Urgency / Pressure", "violet"))
    if MONEY_RE.search(message):
        tags.append(("Money / Prize Mention", "amber"))
    if label == 1:
        tags.append(("Model-Flagged Pattern", "red"))
    return tags


def message_tone(message: str) -> str:
    return "Urgent" if URGENCY_RE.search(message) else "Neutral"


def dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    return fig, ax


def donut_html(spam_pct: float, ham_pct: float, size=150) -> str:
    spam_pct = max(0.0, min(100.0, spam_pct))
    deg = spam_pct * 3.6
    main_pct = spam_pct if spam_pct >= ham_pct else ham_pct
    main_label = "Spam" if spam_pct >= ham_pct else "Not Spam"
    return f"""
    <div class="donut-wrap">
        <div class="donut" style="background: conic-gradient(var(--threat) 0deg {deg}deg, var(--signal) {deg}deg 360deg);">
            <div class="donut-inner">
                <div class="pct">{main_pct:.1f}%</div>
                <div class="lab">{main_label}</div>
            </div>
        </div>
        <div class="donut-legend">
            <span><span class="dot" style="background:var(--threat)"></span>Spam ({spam_pct:.1f}%)</span>
            <span><span class="dot" style="background:var(--signal)"></span>Not Spam ({ham_pct:.1f}%)</span>
        </div>
    </div>
    """


def verdict_html(label: int, confidence) -> str:
    conf = confidence if confidence is not None else 0
    if label == 1:
        return f"""
        <div class="verdict spam">
            <div class="verdict-ring">🚨</div>
            <div style="flex:1;">
                <div class="verdict-label">SPAM</div>
                <div class="verdict-detail">{conf:.2f}% confidence score · {MODEL_NAME}</div>
                <div class="verdict-bar-track"><div class="verdict-bar-fill" style="width:{conf:.1f}%; background:var(--threat);"></div></div>
            </div>
        </div>
        """
    return f"""
    <div class="verdict ham">
        <div class="verdict-ring">✅</div>
        <div style="flex:1;">
            <div class="verdict-label">NOT SPAM</div>
            <div class="verdict-detail">{conf:.2f}% confidence score · {MODEL_NAME}</div>
            <div class="verdict-bar-track"><div class="verdict-bar-fill" style="width:{conf:.1f}%; background:var(--signal);"></div></div>
        </div>
    </div>
    """


def bar_row_html(icon, label, pct, color_var="var(--violet)", value_txt=None):
    value_txt = value_txt if value_txt is not None else f"{pct:.1f}%"
    pct = max(0.0, min(100.0, pct))
    return f"""
    <div class="bar-row">
        <div class="bicon">{icon}</div>
        <div class="blabel">{label}</div>
        <div class="btrack"><div class="bfill" style="width:{pct}%; background:{color_var};"></div></div>
        <div class="bval">{value_txt}</div>
    </div>
    """


def compute_labeled_metrics(df: pd.DataFrame, msg_col: str, label_col: str):
    label_map = {"spam": 1, "ham": 0, "1": 1, "0": 0, 1: 1, 0: 0, "not spam": 0, "spam ": 1}
    y_true_raw = df[label_col].astype(str).str.strip().str.lower()
    y_true = y_true_raw.map(label_map)
    valid = y_true.notna()
    if valid.sum() == 0:
        return None

    messages = df.loc[valid, msg_col].astype(str).tolist()
    y_true = y_true.loc[valid].astype(int).to_numpy()

    cleaned = [normalize_text(m) for m in messages]
    X = vectorizer.transform(cleaned)
    y_pred = model.predict(X)

    if HAS_DECISION:
        scores = model.decision_function(X)
    elif HAS_PROBA:
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = y_pred.astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # [[TP,FN],[FP,TN]] with spam=1 first
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": cm,
        "fpr": fpr, "tpr": tpr, "auc": roc_auc,
        "n": len(y_true),
    }



defaults = {
    "history": [],
    "message_input": "",
    "page": "Dashboard",
    "batch_df": None,
    "batch_msg_col": None,
    "batch_label_col": None,
    "last_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def set_sample(text):
    st.session_state.message_input = text


def go_to(page):
    st.session_state.page = page


SPAM_SAMPLE = "WINNER!! You have been selected to receive a £1000 cash prize. Call 09061234567 now to claim before it expires!"
HAM_SAMPLE = "Hey, just checking — are we still on for lunch tomorrow at 1pm? Let me know if that works for you."



NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("Detect Spam", "🔍"),
    ("Batch Upload", "📁"),
    ("Analytics", "📊"),
    ("Model Performance", "📈"),
    ("History", "🕘"),
    ("Settings", "⚙️"),
]

with st.sidebar:
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-icon">🛡️</div>
            <div class="brand-title">Spam<span class="accent">Guard AI</span></div>
            <div class="brand-sub">AI Spam Email Detection System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for name, icon in NAV_ITEMS:
        active = st.session_state.page == name
        st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            type="primary" if active else "secondary",
            use_container_width=True,
            on_click=go_to,
            args=(name,),
        )

    st.markdown(
        """
        <div class="sidebar-shield">
            <div class="icon">🛡️</div>
            <div class="sidebar-badge">🟢 Smart Protection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Python · scikit-learn · TF-IDF · Streamlit")


last_was_spam = len(st.session_state.history) > 0 and st.session_state.history[-1]["Prediction"] == "Spam"
bell_dot = '<span class="dot"></span>' if last_was_spam else ""

st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-search">🔎 Search anything… <span class="kbd">⌘K</span></div>
        <div class="topbar-right">
            <div class="topbar-bell">🔔{bell_dot}</div>
            <div class="topbar-avatar">
                <div class="av">AI</div>
                <div class="meta"><div class="name">AI Engineer</div><div class="role">admin@spamguard.ai</div></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



def render_scan_console(full=False):
    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">✉️ Detect Spam</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-caption">Analyze your email or SMS using the trained TF-IDF + '
            f'{MODEL_NAME} model.</div>',
            unsafe_allow_html=True,
        )

        message = st.text_area(
            "message_box", height=200 if full else 170,
            placeholder="Paste your email / SMS here…",
            key="message_input", label_visibility="collapsed",
        )

        if full:
            b1, b2, b3, b4 = st.columns(4)
        else:
            b1, b2 = st.columns(2)
            b3 = b4 = None

        with b1:
            scan_clicked = st.button("🔍 Analyze Email", use_container_width=True)
        with b2:
            st.button("Try spam ↗", use_container_width=True, type="secondary", on_click=set_sample, args=(SPAM_SAMPLE,))
        if full:
            with b3:
                st.button("Try clean ↗", use_container_width=True, type="secondary", on_click=set_sample, args=(HAM_SAMPLE,))
            with b4:
                st.button("Clear", use_container_width=True, type="secondary", on_click=set_sample, args=("",))

        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🛡️ AI Analysis Result</div>', unsafe_allow_html=True)

        if scan_clicked and message.strip() == "":
            st.warning("Enter a message before analyzing.")
        elif message.strip() == "":
            st.markdown(
                '<div class="panel-caption">Run an analysis to see the verdict, '
                'confidence and contributing words.</div>', unsafe_allow_html=True,
            )
        else:
            label, confidence, raw_score, vec = score_message(message)
            spam_tok, ham_tok = message_top_tokens(vec)

            st.markdown(verdict_html(label, confidence), unsafe_allow_html=True)

            tags = reason_tags(message, label)
            if tags:
                st.markdown('<div class="panel-caption" style="margin-top:8px;">Why this verdict?</div>', unsafe_allow_html=True)
                chips = "".join(f'<span class="tag {cls}">{txt}</span>' for txt, cls in tags)
                st.markdown(chips, unsafe_allow_html=True)

            if full and (spam_tok or ham_tok):
                st.markdown('<div class="panel-caption" style="margin-top:10px;">Top contributing words</div>', unsafe_allow_html=True)
                chips = ""
                for word, w in spam_tok:
                    chips += f'<span class="tag red">▲ {word} <span style="opacity:.6">{w:+.2f}</span></span>'
                for word, w in ham_tok:
                    chips += f'<span class="tag green">▼ {word} <span style="opacity:.6">{w:+.2f}</span></span>'
                st.markdown(chips, unsafe_allow_html=True)

            if not HAS_PROBA:
                st.caption("Confidence reflects the SVM's decision margin, not a calibrated probability.")

            st.session_state.last_result = {
                "message": message, "label": label, "confidence": confidence,
                "spam_tok": spam_tok, "ham_tok": ham_tok,
            }

            if scan_clicked:
                st.session_state.history.append({
                    "Message": message[:160] + ("…" if len(message) > 160 else ""),
                    "Prediction": "Spam" if label == 1 else "Not Spam",
                    "Confidence": round(confidence, 2) if confidence is not None else None,
                })

        st.markdown('</div>', unsafe_allow_html=True)

    return message if message.strip() else None


def render_probability_and_details(message):
    col_donut, col_details = st.columns(2, gap="large")
    label, confidence, raw_score, vec = score_message(message)
    spam_pct = confidence if label == 1 else 100 - confidence
    ham_pct = 100 - spam_pct

    with col_donut:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Prediction Probability</div>', unsafe_allow_html=True)
        st.markdown(donut_html(spam_pct, ham_pct), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_details:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Email Details</div>', unsafe_allow_html=True)
        spam_tok, _ = message_top_tokens(vec)
        n_links = len(URL_RE.findall(message))
        rows = [
            ("Word Count", len(message.split())),
            ("Character Count", len(message)),
            ("Links Found", n_links),
            ("Spam Keywords", len(spam_tok)),
            ("Tone", message_tone(message)),
            ("Language", "English"),
        ]
        html = "".join(f'<div class="detail-row"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in rows)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_kpis():
    total_scans = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h["Prediction"] == "Spam")
    confidences = [h["Confidence"] for h in st.session_state.history if h["Confidence"] is not None]
    avg_conf = f"{np.mean(confidences):.1f}%" if confidences else "—"

    cards = [
        ("Total Scans", total_scans, "this session", "📧", "rgba(124,108,246,0.18)", VIOLET),
        ("Spam Detected", spam_count, "this session", "🚫", "rgba(251,69,112,0.18)", THREAT),
        ("Avg. Confidence", avg_conf, "across scans", "🎯", "rgba(77,141,255,0.18)", BLUE),
        (MODEL_NAME, f"{VOCAB_SIZE:,} features", "Active", "🧊", "rgba(45,212,191,0.18)", SIGNAL),
    ]
    cols = st.columns(4)
    for col, (label, value, caption, icon, bg, fg) in zip(cols, cards):
        badge = f'<div class="kpi-badge">{caption}</div>' if caption == "Active" else f'<div class="kpi-caption">{caption}</div>'
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-top">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-icon" style="background:{bg}; color:{fg};">{icon}</div>
                    </div>
                    <div class="kpi-value">{value}</div>
                    {badge}
                </div>
                """,
                unsafe_allow_html=True,
            )




if st.session_state.page == "Dashboard":
    render_kpis()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    msg = render_scan_console(full=False)
    if msg:
        render_probability_and_details(msg)



elif st.session_state.page == "Detect Spam":
    msg = render_scan_console(full=True)
    if msg:
        if msg.strip():
            chars, words, lines = len(msg), len(msg.split()), len(msg.split("\n"))
            m1, m2, m3 = st.columns(3)
            m1.metric("Characters", chars)
            m2.metric("Words", words)
            m3.metric("Lines", lines)
        render_probability_and_details(msg)



elif st.session_state.page == "Batch Upload":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📁 Batch Scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Upload a CSV with a <code>message</code> column, or a .txt file. '
        'Add an optional <code>label</code> column (spam/ham) to unlock real accuracy metrics on the '
        '<b>Model Performance</b> page.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            label, confidence, _, _ = score_message(text)
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Result</div>', unsafe_allow_html=True)
            st.markdown(verdict_html(label, confidence), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Couldn't read this CSV: {e}")
                df = None

            if df is not None:
                cols_lower = {c.lower(): c for c in df.columns}
                msg_col = cols_lower.get("message")
                label_col = cols_lower.get("label")

                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Dataset Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if msg_col is None:
                    st.error("CSV must contain a column named 'message'.")
                else:
                    predictions, confidences = [], []
                    for m in df[msg_col]:
                        lbl, conf, _, _ = score_message(str(m))
                        predictions.append("Spam" if lbl == 1 else "Not Spam")
                        confidences.append(round(conf, 2) if conf is not None else 0)

                    df["Prediction"] = predictions
                    df["Confidence"] = confidences
                    st.session_state.batch_df = df
                    st.session_state.batch_msg_col = msg_col
                    st.session_state.batch_label_col = label_col

                    spam_count = (df["Prediction"] == "Spam").sum()
                    ham_count = (df["Prediction"] == "Not Spam").sum()

                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-title">Results</div>', unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Messages", len(df))
                    m2.metric("🚫 Spam", int(spam_count))
                    m3.metric("✅ Not Spam", int(ham_count))
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False)
                    st.download_button("⬇ Download predictions (.csv)", csv, "predictions.csv", "text/csv")
                    if label_col:
                        st.success(f"Detected a '{label_col}' column — head to **Model Performance** for real accuracy metrics.")
                    st.markdown('</div>', unsafe_allow_html=True)

                    chart1, chart2 = st.columns(2)
                    with chart1:
                        st.markdown('<div class="panel">', unsafe_allow_html=True)
                        st.markdown('<div class="panel-title">Distribution</div>', unsafe_allow_html=True)
                        fig, ax = dark_fig((4.5, 4.5))
                        ax.pie([spam_count, ham_count], labels=["Spam", "Not Spam"], autopct="%1.1f%%",
                               colors=[THREAT, SIGNAL], startangle=90,
                               textprops={"color": TEXT, "fontsize": 11},
                               wedgeprops={"edgecolor": PANEL, "linewidth": 2})
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with chart2:
                        st.markdown('<div class="panel">', unsafe_allow_html=True)
                        st.markdown('<div class="panel-title">Counts</div>', unsafe_allow_html=True)
                        fig2, ax2 = dark_fig((4.5, 4.5))
                        ax2.bar(["Spam", "Not Spam"], [spam_count, ham_count], color=[THREAT, SIGNAL])
                        ax2.set_ylabel("Count", color=MUTED)
                        ax2.tick_params(colors=MUTED)
                        for spine in ax2.spines.values():
                            spine.set_color(LINE)
                        st.pyplot(fig2)
                        st.markdown('</div>', unsafe_allow_html=True)



elif st.session_state.page == "Analytics":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">☁ Word Cloud</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Based on your last batch upload, or the message currently in '
        'Detect Spam.</div>', unsafe_allow_html=True,
    )

    def render_wordcloud(text, colormap):
        if not text.strip():
            return
        wc = WordCloud(width=900, height=380, background_color=PANEL, colormap=colormap).generate(text)
        fig, ax = dark_fig((9, 3.8))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    bdf = st.session_state.batch_df
    if bdf is not None and "Prediction" in bdf.columns:
        msg_col = st.session_state.batch_msg_col
        spam_text = " ".join(bdf.loc[bdf["Prediction"] == "Spam", msg_col].astype(str))
        ham_text = " ".join(bdf.loc[bdf["Prediction"] == "Not Spam", msg_col].astype(str))
        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("**🚫 Spam vocabulary**")
            render_wordcloud(spam_text, "autumn")
        with wc2:
            st.markdown("**✅ Not-spam vocabulary**")
            render_wordcloud(ham_text, "winter")
    elif st.session_state.message_input.strip():
        st.markdown("**Current message**")
        render_wordcloud(st.session_state.message_input, "cool")
    else:
        st.info("Nothing to visualize yet — scan a message or upload a batch CSV first.")

    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.page == "Model Performance":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🧠 How a message gets classified</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Mirrors the exact pipeline trained in <code>main.ipynb</code>.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="pipeline">
            <div class="pipe-step"><div class="pn">STEP 01</div><div class="pt">Raw message</div><div class="pd">Email / SMS text, untouched</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step"><div class="pn">STEP 02</div><div class="pt">Light normalization</div><div class="pd">Strip URLs &amp; whitespace only</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step"><div class="pn">STEP 03</div><div class="pt">TF-IDF vectorizer</div><div class="pd">{VOCAB_SIZE:,} features, English stop-words removed</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step"><div class="pn">STEP 04</div><div class="pt">{MODEL_NAME}</div><div class="pd">Linear boundary over sparse TF-IDF vectors</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step"><div class="pn">STEP 05</div><div class="pt">Verdict</div><div class="pd">Spam / Not spam + confidence</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📚 Training Data</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:13px; line-height:1.9; color:var(--text);">
            dataset&nbsp;size = <span style="color:var(--signal)">{TRAIN_FACTS['dataset_size']:,}</span> messages<br>
            train&nbsp;/&nbsp;test = <span style="color:var(--signal)">{TRAIN_FACTS['train_size']:,}</span> / <span style="color:var(--signal)">{TRAIN_FACTS['test_size']:,}</span> (80/20)<br>
            vectorizer = <span style="color:var(--signal)">{TRAIN_FACTS['vectorizer']}</span><br>
            vocabulary = <span style="color:var(--signal)">{VOCAB_SIZE:,}</span> features
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">⚖️ Models Benchmarked</div>', unsafe_allow_html=True)
        for c in TRAIN_FACTS["candidates"]:
            picked = c == "Linear SVM" and MODEL_NAME == "LinearSVC"
            marker = "🟢" if picked else "⚪"
            st.markdown(f"{marker} **{c}**{' — deployed' if picked else ''}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🏆 Top Spam Indicators</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Strongest spam-pushing tokens, read directly from the trained '
        f'{MODEL_NAME} coefficients (model-wide, not message-specific).</div>', unsafe_allow_html=True,
    )
    tokens = global_top_tokens(8)
    if tokens:
        max_coef = max(w for _, w in tokens)
        rows = "".join(
            bar_row_html("▲", word, (w / max_coef) * 100, "var(--threat)", f"{w:+.2f}")
            for word, w in tokens
        )
        st.markdown(rows, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    bdf = st.session_state.batch_df
    label_col = st.session_state.batch_label_col
    metrics = None
    if bdf is not None and label_col is not None:
        metrics = compute_labeled_metrics(bdf, st.session_state.batch_msg_col, label_col)

    if metrics is None:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📈 Accuracy, Confusion Matrix &amp; ROC</div>', unsafe_allow_html=True)
        st.info(
            "Upload a batch CSV with a `message` column **and** a `label` column "
            "(spam/ham) on the **Batch Upload** page to compute real accuracy, "
            "precision, recall, F1, a confusion matrix and an ROC curve on your own data."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">📈 Performance on uploaded data <span class="pill low">{metrics["n"]} labeled rows</span></div>', unsafe_allow_html=True)
        rows = (
            bar_row_html("🎯", "Accuracy", metrics["accuracy"] * 100, "var(--violet)")
            + bar_row_html("🛡️", "Precision", metrics["precision"] * 100, "var(--violet)")
            + bar_row_html("🔁", "Recall", metrics["recall"] * 100, "var(--violet)")
            + bar_row_html("⚖️", "F1 Score", metrics["f1"] * 100, "var(--violet)")
        )
        st.markdown(rows, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col_cm, col_roc = st.columns(2, gap="large")
        with col_cm:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = metrics["cm"]  # [[TP, FN], [FP, TN]]
            tp, fn = cm[0]
            fp, tn = cm[1]
            st.markdown(
                f"""
                <div class="cm-grid">
                    <div></div>
                    <div class="cm-head">Actual: Spam</div>
                    <div class="cm-head">Actual: Ham</div>
                    <div class="cm-rowlabel">Pred: Spam</div>
                    <div class="cm-cell" style="background:rgba(251,69,112,0.22); color:var(--threat);">{tp}</div>
                    <div class="cm-cell" style="background:var(--panel-2); color:var(--text);">{fp}</div>
                    <div class="cm-rowlabel">Pred: Ham</div>
                    <div class="cm-cell" style="background:var(--panel-2); color:var(--text);">{fn}</div>
                    <div class="cm-cell" style="background:rgba(45,212,191,0.22); color:var(--signal);">{tn}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_roc:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">ROC Curve</div>', unsafe_allow_html=True)
            fig, ax = dark_fig((5, 4.2))
            ax.plot(metrics["fpr"], metrics["tpr"], color=VIOLET, linewidth=2.2)
            ax.plot([0, 1], [0, 1], color=LINE, linestyle="--", linewidth=1)
            ax.set_xlabel("False Positive Rate", color=MUTED)
            ax.set_ylabel("True Positive Rate", color=MUTED)
            ax.tick_params(colors=MUTED)
            for spine in ax.spines.values():
                spine.set_color(LINE)
            ax.text(0.55, 0.08, f"AUC = {metrics['auc']:.3f}", color=TEXT, fontsize=11)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)



elif st.session_state.page == "History":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🕘 Scan History</div>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.markdown('<div class="panel-caption">No scans yet this session — results will appear here.</div>', unsafe_allow_html=True)
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            csv = history_df.to_csv(index=False)
            st.download_button("⬇ Download history (.csv)", csv, "history.csv", "text/csv")
        with c2:
            if st.button("🗑 Clear history", type="secondary"):
                st.session_state.history = []
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)



elif st.session_state.page == "Settings":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="detail-row"><span class="k">Classifier</span><span class="v">{MODEL_NAME}</span></div>
        <div class="detail-row"><span class="k">Vocabulary size</span><span class="v">{VOCAB_SIZE:,}</span></div>
        <div class="detail-row"><span class="k">Exposes probabilities</span><span class="v">{HAS_PROBA}</span></div>
        <div class="detail-row"><span class="k">App</span><span class="v">SpamGuard AI</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Session data</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-caption">Clears scan history and any uploaded batch file from this session.</div>', unsafe_allow_html=True)
    if st.button("🗑 Reset session data", type="secondary"):
        st.session_state.history = []
        st.session_state.batch_df = None
        st.session_state.batch_msg_col = None
        st.session_state.batch_label_col = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    """
    <div class="footer-line">
        🛡️ <b>SpamGuard AI</b> — AI Spam Email Detection System &nbsp;·&nbsp;
        Python · scikit-learn · TF-IDF · Streamlit &nbsp;·&nbsp;
        Developed by <b>Aryan Mehta</b>
    </div>
    """,
    unsafe_allow_html=True,
)

