"""
app.py  -  MedQuery AI (Consumer Edition)
Plain-English health research assistant powered by PubMed + Mistral.
Run with: streamlit run app.py
"""

import streamlit as st
import os
import time
import re
from dotenv import load_dotenv
from rag.pubmed_fetcher import fetch_papers_for_topic
from rag.embedder import MedEmbedder
from rag.llm import get_llm

load_dotenv()

st.set_page_config(
    page_title="MedQuery AI - Health Questions, Real Answers",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff;
    font-family: 'Inter', sans-serif;
    color: #111111;
  }
  [data-testid="stSidebar"] { display: none; }
  [data-testid="stHeader"]  { background: transparent; }
  .block-container { padding-top: 2rem; max-width: 760px; }

  /* Hero */
  .hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
  .hero-icon { font-size: 3.5rem; line-height: 1; margin-bottom: 0.5rem; }
  .hero h1 {
    font-size: 2.2rem; font-weight: 700; color: #000000;
    margin: 0 0 0.4rem; letter-spacing: -0.02em;
  }
  .hero p { color: #555555; font-size: 1rem; margin: 0; }

  /* Text area */
  .stTextArea > div > div > textarea {
    background: #f5f5f5 !important;
    color: #111111 !important;
    caret-color: #000000 !important;
    border: 2px solid #000000 !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    box-shadow: none !important;
  }
  .stTextArea > div > div > textarea::placeholder {
    color: #999999 !important;
  }
  .stTextArea > div > div > textarea:focus {
    box-shadow: 0 0 0 3px rgba(0,0,0,0.12) !important;
  }

  /* Buttons */
  .stButton > button {
    background: #000000 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
  }
  .stButton > button:hover {
    background: #222222 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
  }
  .stButton > button:disabled {
    background: #cccccc !important;
    color: #888888 !important;
    box-shadow: none !important;
    transform: none !important;
  }

  /* Answer card */
  .answer-card {
    background: #000000;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    border-left: 4px solid #ffffff;
    line-height: 1.85;
    font-size: 1rem;
    color: #ffffff;
  }
  .answer-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #aaaaaa;
    margin-bottom: 0.8rem;
  }

  /* Alert box */
  [data-testid="stAlert"] {
    background: #f5f5f5 !important;
    border: 1px solid #cccccc !important;
    color: #444444 !important;
    border-radius: 10px !important;
  }

  /* Source cards */
  .source-card {
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
  }
  .source-title { font-size: 0.88rem; font-weight: 500; color: #111111; margin-bottom: 0.2rem; }
  .source-meta  { font-size: 0.75rem; color: #888888; }

  /* Chat bubbles */
  .q-bubble {
    background: #f0f0f0;
    border-radius: 12px 12px 12px 2px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.4rem;
    font-size: 0.92rem;
    color: #111111;
    max-width: 85%;
  }
  .a-bubble {
    background: #ffffff;
    border: 1px solid #dddddd;
    border-radius: 12px 12px 2px 12px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 1.2rem;
    font-size: 0.9rem;
    color: #333333;
    line-height: 1.7;
    max-width: 92%;
    margin-left: auto;
  }

  /* Footer */
  .disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: #999999;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #eeeeee;
  }

  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Session state
if "embedder" not in st.session_state:
    st.session_state.embedder = MedEmbedder()
if "llm" not in st.session_state:
    try:
        st.session_state.llm = get_llm()
        st.session_state.llm_ready = True
    except EnvironmentError:
        st.session_state.llm_ready = False
if "history"  not in st.session_state:
    st.session_state.history = []
if "prefill"  not in st.session_state:
    st.session_state.prefill = ""

embedder: MedEmbedder = st.session_state.embedder

# Topic auto-detection
TOPIC_MAP = {
    "diabet":        "diabetes symptoms treatment prevention genetics",
    "insulin":       "diabetes insulin treatment",
    "blood sugar":   "diabetes blood glucose management",
    "cancer":        "cancer treatment immunotherapy symptoms",
    "tumor":         "cancer tumor treatment",
    "heart":         "cardiovascular disease prevention treatment",
    "cardiac":       "cardiovascular disease prevention treatment",
    "cholesterol":   "cholesterol management cardiovascular",
    "alzheimer":     "Alzheimer disease symptoms treatment",
    "dementia":      "dementia Alzheimer treatment",
    "covid":         "COVID-19 symptoms long-term effects",
    "coronavirus":   "COVID-19 symptoms long-term effects",
    "sleep":         "sleep disorders insomnia treatment",
    "insomnia":      "sleep disorders insomnia treatment",
    "anxiety":       "anxiety depression mental health treatment",
    "depress":       "anxiety depression mental health treatment",
    "mental health": "mental health anxiety depression treatment",
    "blood pressure":"hypertension blood pressure management",
    "hypertension":  "hypertension blood pressure management",
    "obes":          "obesity weight management treatment",
    "weight":        "obesity weight management diet",
    "asthma":        "asthma treatment management",
    "arthrit":       "arthritis joint pain treatment",
    "joint":         "arthritis joint pain treatment",
    "migraine":      "migraine headache treatment prevention",
    "headache":      "migraine headache treatment",
    "thyroid":       "thyroid disorders treatment",
    "kidney":        "kidney disease treatment prevention",
    "liver":         "liver disease hepatitis treatment",
    "stroke":        "stroke prevention treatment recovery",
    "vitamin":       "vitamin deficiency nutrition health",
    "immune":        "immune system health treatment",
    "gene":          "genetics hereditary disease genetic risk",
    "heredit":       "hereditary genetic disease risk",
    "born":          "genetic hereditary disease congenital",
    "birth":         "genetic congenital hereditary disease",
    "inherit":       "hereditary genetics disease risk",
    "pregnan":       "pregnancy health complications",
    "infect":        "infection treatment antibiotics",
    "bacteria":      "bacterial infection antibiotic resistance",
    "virus":         "viral infection treatment",
    "allerg":        "allergy treatment immunology",
    "skin":          "skin disease dermatology treatment",
    "bone":          "bone health osteoporosis treatment",
    "lung":          "lung disease pulmonary treatment",
    "breath":        "respiratory lung disease treatment",
}

def detect_topic(question: str) -> str:
    q = question.lower()
    for keyword, topic in TOPIC_MAP.items():
        if keyword in q:
            return topic
    words = [w for w in re.sub(r'[^\w\s]', '', question).split()
             if len(w) > 3 and w.lower() not in
             {"what","does","have","this","that","with","from","just","because","your",
              "about","some","when","will","they","them","there","their","were","been",
              "into","than","then","also","only","more","very","over","after","before",
              "would","could","should","which","while","these","those","other","such",
              "even","much","many","like"}]
    return " ".join(words[:5]) + " medical health research"

def ensure_papers(question: str):
    topic = detect_topic(question)
    papers = fetch_papers_for_topic(topic, max_results=20)
    if papers:
        embedder.add_papers(papers)

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-icon">🩺</div>
  <h1>MedQuery AI</h1>
  <p>Ask any health question - get clear, research-backed answers in plain English</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.llm_ready:
    st.error("Setup incomplete - check your .env file has MISTRAL_API_KEY set.", icon="⚠️")
    st.stop()

# Suggested questions
suggestions = [
    "What are early signs of diabetes?",
    "How can I lower blood pressure naturally?",
    "What causes high cholesterol?",
    "How does sleep affect heart health?",
    "What are symptoms of anxiety?",
    "Can diet help prevent cancer?",
]

cols = st.columns(3)
for i, s in enumerate(suggestions):
    with cols[i % 3]:
        if st.button(s, key=f"chip_{i}"):
            st.session_state.prefill = s
            st.rerun()

# Question input
question = st.text_area(
    "",
    value=st.session_state.prefill,
    placeholder="e.g. What are the early warning signs of diabetes?",
    height=110,
    label_visibility="collapsed",
)

ask_btn = st.button("Get Answer", disabled=not question.strip())

# Answer generation
if ask_btn and question.strip():
    st.session_state.prefill = ""
    with st.spinner("Searching research and writing your answer..."):
        t0 = time.time()
        ensure_papers(question)
        relevant = embedder.query(question, n_results=5)
        if not relevant:
            st.warning("Couldn't find research on that topic. Try rephrasing your question.")
        else:
            answer  = st.session_state.llm.answer(question, relevant, mode="simple")
            elapsed = round(time.time() - t0, 1)
            st.session_state.history.append({
                "question": question,
                "answer":   answer,
                "papers":   relevant,
                "elapsed":  elapsed,
            })

# Display latest answer
if st.session_state.history:
    latest = st.session_state.history[-1]

    st.markdown(f"""
<div class="answer-card">
  <div class="answer-label">Answer - based on {len(latest['papers'])} research studies</div>
  {latest['answer']}
</div>
""", unsafe_allow_html=True)

    with st.expander("View research sources"):
        for p in latest["papers"]:
            st.markdown(f"""
<div class="source-card">
  <div class="source-title">{p['title']}</div>
  <div class="source-meta">{p['journal']} - {p['year']} -
    <a href="{p['url']}" target="_blank" style="color:#000000;font-weight:500;">Read full study</a>
  </div>
</div>
""", unsafe_allow_html=True)

    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("**Previous questions**")
        for item in reversed(st.session_state.history[:-1]):
            st.markdown(f'<div class="q-bubble">🙋 {item["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="a-bubble">{item["answer"]}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="disclaimer">
  MedQuery AI provides general health information based on published medical research.<br/>
  It is <strong>not a substitute for professional medical advice, diagnosis, or treatment.</strong><br/>
  Always consult a qualified healthcare provider for personal medical concerns.<br/><br/>
  <span style="color:#cccccc;">Built by Sri Vardhini Venna</span>
</div>
""", unsafe_allow_html=True)