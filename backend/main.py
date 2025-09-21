import os
import tempfile
import shutil
import json
import sqlite3
from typing import List, Optional
from math import sqrt

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import pdfplumber
import docx
from rapidfuzz import fuzz

# -----------------------
# Load environment
# -----------------------
load_dotenv()

try:
    import openai
except ImportError:
    openai = None

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", 8000))
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")

if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY
    DEMO_MODE = False

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="Resume Relevance Backend (MVP)")

# -----------------------
# Serve frontend
# -----------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_root():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="index.html not found")

# -----------------------
# Database setup
# -----------------------
DB_FILE = os.path.join(os.path.dirname(__file__), "evaluations.db")


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
# Models
# -----------------------
class EvalResult(BaseModel):
    job_title: Optional[str]
    hard_score: int
    semantic_score: int
    final_score: int
    missing: List[str]
    verdict: str
    suggestions: List[str]

# -----------------------
# File parsing helpers
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


def parse_uploaded_file(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename)[1].lower()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, upload.filename)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    text = ""
    try:
        if suffix in (".txt",):
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read()
        elif suffix in (".pdf",):
            text = extract_text_from_pdf(path)
        elif suffix in (".docx", ".doc"):
            text = extract_text_from_docx(path)
        else:
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read()
    finally:
        shutil.rmtree(tmpdir)
    return text

# -----------------------
# Skills extraction
# -----------------------
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
    final = []
    for s in skills:
        s2 = " ".join(s.split())
        if s2 and s2.lower() not in [x.lower() for x in final]:
            final.append(s2)
    return final

# -----------------------
# Hard matching
# -----------------------
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

# -----------------------
# Embeddings & cosine
# -----------------------
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
    else:
        if v1 is None or v2 is None:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        a2 = sqrt(sum(a * a for a in v1))
        b2 = sqrt(sum(b * b for b in v2))
        if a2 == 0 or b2 == 0:
            return 0.0
        return dot / (a2 * b2)

# -----------------------
# Suggestions
# -----------------------
def generate_suggestions_with_llm(jd_text: str, resume_text: str, missing: List[str], demo_title: str = None):
    if not DEMO_MODE and openai:
        prompt = f"""
You are an assistant that generates concise resume improvement suggestions.

Job description:
{jd_text}

Candidate resume:
{resume_text}

Missing skills: {', '.join(missing) if missing else 'None'}
"""
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You provide actionable resume feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2
        )
        text = resp['choices'][0]['message']['content'].strip()
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        return parts if parts else [text]
    else:
        s = []
        if missing:
            s.append(f"Add or emphasize these skills: {', '.join(missing[:8])}.")
            for m in missing[:6]:
                s.append(f"Show a project: 'Built X using {m} — achieved Y'.")
        else:
            s.append("Skills look covered. Add measurable impact (numbers).")
        if demo_title:
            s.append(f"Tailor your resume summary to mention: {demo_title}.")
        return s

# -----------------------
# API Endpoints
# -----------------------
@app.post("/evaluate", response_model=EvalResult)
async def evaluate(
    jd_text: Optional[str] = Form(None),
    resume_text: Optional[str] = Form(None),
    jd_file: UploadFile = File(None),
    resume_file: UploadFile = File(None),
    candidate_name: Optional[str] = Form(None)
):
    if jd_file:
        jd_text = parse_uploaded_file(jd_file)
    if resume_file:
        resume_text = parse_uploaded_file(resume_file)

    if not jd_text or not resume_text:
        raise HTTPException(status_code=400, detail="Provide jd_text and resume_text (or upload files).")

    # Extract job title
    job_title = None
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    if lines:
        first = lines[0]
        if first.lower().startswith("role:") or first.lower().startswith("title:"):
            job_title = first.split(":", 1)[1].strip()
        else:
            job_title = first[:120]

    # Hard + semantic match
    jd_skills = extract_skills_from_jd(jd_text)
    hard = hard_match(jd_skills, resume_text)

    emb_jd = get_embedding(jd_text)
    emb_resume = get_embedding(resume_text)
    sem = int(round(cosine(emb_jd, emb_resume) * 100))

    final = int(round(0.7 * hard["score"] + 0.3 * sem))
    verdict = "High" if final >= 75 else ("Medium" if final >= 50 else "Low")

    suggestions = generate_suggestions_with_llm(jd_text, resume_text, hard["missing"], job_title)

    # Save to DB
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO evaluations (job_title, jd_text, candidate_name, resume_text, hard_score, semantic_score, final_score, missing, verdict, suggestions, raw)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        job_title or "",
        jd_text[:2000],
        candidate_name or "",
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

    return {
        "job_title": job_title,
        "hard_score": hard["score"],
        "semantic_score": sem,
        "final_score": final,
        "missing": hard["missing"],
        "verdict": verdict,
        "suggestions": suggestions
    }


@app.get("/evaluations")
def list_evals(limit: int = 100):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, job_title, candidate_name, hard_score, semantic_score, final_score, verdict, missing, suggestions 
        FROM evaluations 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "job_title": r[1],
            "candidate_name": r[2],
            "hard_score": r[3],
            "semantic_score": r[4],
            "final_score": r[5],
            "verdict": r[6],
            "missing": json.loads(r[7]) if r[7] else [],
            "suggestions": json.loads(r[8]) if r[8] else []
        })
    return JSONResponse(content=out)
