import streamlit as st
import openai
import os
import pdfplumber
import docx
from rapidfuzz import fuzz

# Load API key from environment (set in Streamlit secrets for deployment)
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Resume Relevance", layout="wide")
st.title("📄 Resume Relevance Checker")

# Job Description input
jd = st.text_area("Paste Job Description (JD):", height=150)

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX):", type=["pdf", "docx"])

resume_text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            resume_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        resume_text = "\n".join([para.text for para in doc.paragraphs])

# Candidate name
candidate_name = st.text_input("Candidate Name:")

# Evaluate button
if st.button("Evaluate Resume"):
    if not jd or not resume_text:
        st.warning("⚠️ Please provide both a JD and a Resume.")
    else:
        # Fuzzy matching
        fuzz_score = fuzz.token_set_ratio(jd, resume_text)

        st.subheader("🔎 Match Analysis")
        st.write(f"**Fuzzy Match Score:** {fuzz_score}%")

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant that evaluates resumes against job descriptions."},
                    {"role": "user", "content": f"Job Description:\n{jd}\n\nResume:\n{resume_text}\n\nCandidate: {candidate_name}"}
                ]
            )
            st.subheader("📝 AI Evaluation")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"❌ Error: {e}")
