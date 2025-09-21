import os
import json
import sqlite3
import tempfile
import shutil
import pdfplumber
import docx
import re
from math import sqrt
from typing import List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import fuzz
from dotenv import load_dotenv

# -----------------------
# Load environment
# -----------------------
load_dotenv()

try:
    import openai
except ImportError:
    openai = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")

if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY
    DEMO_MODE = False

# -----------------------
# Database
# -----------------------
DB_FILE = "evaluations.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT,
        jd_text TEXT,
        candidate_name TEXT,
        resume_text TEXT,
        hard_score INTEGER,
        semantic_score INTEGER,
        final_score INTEGER,
        missing TEXT,
        verdict TEXT,
        suggestions TEXT,
        raw JSON
    )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------
# Helpers
# -----------------------
def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text)

def extract_text_from_pdf(file_path: str) -> str:
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return "\n".join(texts)

def parse_uploaded_file(upload) -> str:
    suffix = os.path.splitext(upload.name)[1].lower()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, upload.name)
    with open(path, "wb") as f:
        f.write(upload.read())
    text = ""
    try:
        if suffix == ".pdf":
            text = extract_text_from_pdf(path)
        elif suffix in (".docx", ".doc"):
            text = extract_text_from_docx(path)
        else:
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read()
    finally:
        shutil.rmtree(tmpdir)
    return text

def extract_skills_from_jd(jd_text: str) -> List[str]:
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    skills = []
    for l in lines:
        if "," in l or "must" in l.lower() or "good" in l.lower() or "-" in l[:2]:
            cleaned = l.replace("Must-have:", "").replace("Good-to-have:", "")
            parts = [p.strip().strip("•-") for p in cleaned.split(",") if p.strip()]
            for p in parts:
                if len(p) < 100:
                    skills.append(p)
    if not skills:
        tokens = set()
        for l in lines[:6]:
            for t in l.split():
                if len(t) > 1 and any(c.isalpha() for c in t):
                    tokens.add(t.strip(",:;()."))
        skills = list(tokens)[:30]
    return skills

def hard_match(jd_skills: List[str], resume_text: str):
    resume_l = resume_text.lower()
    matches = 0
    missing = []
    for skill in jd_skills:
        s = skill.lower()
        if s in resume_l:
            matches += 1
            continue
        score = fuzz.partial_ratio(s, resume_l)
        if score > 70:
            matches += 1
        else:
            missing.append(skill)
    hard_score = int(round((matches / len(jd_skills) * 100))) if jd_skills else 0
    return {"score": hard_score, "matches": matches, "total": len(jd_skills), "missing": missing}

def get_embedding(text: str):
    if not DEMO_MODE and openai:
        resp = openai.Embedding.create(model=EMBED_MODEL, input=text)
        return resp["data"][0]["embedding"]
    else:
        tokens = [t for t in text.lower().split() if len(t) > 2][:512]
        vec = {}
        for t in tokens:
            vec[t] = vec.get(t, 0) + 1
        return vec

def cosine(v1, v2):
    if isinstance(v1, dict) and isinstance(v2, dict):
        dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in v1)
        a2 = sum(v * v for v in v1.values())
        b2 = sum(v * v for v in v2.values())
        if a2 == 0 or b2 == 0:
            return 0.0
        return dot / (sqrt(a2) * sqrt(b2))
    return 0.0

def generate_suggestions(jd_text: str, resume_text: str, missing: List[str], job_title: str = None):
    if not DEMO_MODE and openai:
        prompt = f"""
Job description:
{jd_text}

Candidate resume:
{resume_text}

Missing skills: {', '.join(missing) if missing else 'None'}
"""
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "Provide actionable resume feedback."},
                      {"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return [resp["choices"][0]["message"]["content"].strip()]
    else:
        if missing:
            return [f"Add or emphasize these skills: {', '.join(missing[:8])}."]
        return ["Resume covers required skills. Add measurable achievements."]

# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="Resume Relevance", layout="wide")
st.title("📄 Resume Relevance Checker")

page = st.sidebar.radio("Navigate", ["Evaluate", "Dashboard"])

if page == "Evaluate":
    st.subheader("Upload JD and Resume")

    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    candidate_name = st.text_input("Candidate Name")

    if st.button("Evaluate"):
        if not jd_file or not resume_file:
            st.error("Please upload both JD and Resume")
        else:
            jd_text = parse_uploaded_file(jd_file)
            resume_text = parse_uploaded_file(resume_file)

            jd_skills = extract_skills_from_jd(jd_text)
            hard = hard_match(jd_skills, resume_text)

            emb_jd = get_embedding(jd_text)
            emb_resume = get_embedding(resume_text)
            sem = int(round(cosine(emb_jd, emb_resume) * 100))

            final = int(round(0.7 * hard["score"] + 0.3 * sem))
            verdict = "High" if final >= 75 else ("Medium" if final >= 50 else "Low")
            suggestions = generate_suggestions(jd_text, resume_text, hard["missing"])

            # Save to DB
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO evaluations (job_title, jd_text, candidate_name, resume_text, hard_score, semantic_score, final_score, missing, verdict, suggestions, raw)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                "JD",
                jd_text[:2000],
                candidate_name,
                resume_text[:5000],
                hard["score"],
                sem,
                final,
                json.dumps(hard["missing"]),
                verdict,
                json.dumps(suggestions),
                json.dumps({"jd_skills": jd_skills})
            ))
            conn.commit()
            conn.close()

            st.success(f"Verdict: {verdict}")
            st.write(f"Hard Score: {hard['score']}% | Semantic Score: {sem}% | Final Score: {final}%")
            st.write("Missing skills:", ", ".join(hard["missing"]) or "None")
            st.write("Suggestions:")
            for s in suggestions:
                st.write("-", s)

            # Chart
            fig, ax = plt.subplots()
            ax.bar(["Hard", "Semantic", "Final"], [hard["score"], sem, final], color=["#4CAF50", "#FFC107", "#2196F3"])
            st.pyplot(fig)

elif page == "Dashboard":
    st.subheader("Past Evaluations")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("No evaluations yet.")
