import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.title("Resume Analyzer AI")

st.subheader("Upload your Resume")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_desc = st.text_area("Paste Job Description")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

    if st.button("Analyze Resume"):

        text = [resume_text, job_desc]

        cv = CountVectorizer()
        matrix = cv.fit_transform(text)

        similarity = cosine_similarity(matrix)[0][1]

        score = round(similarity * 100,2)

        st.write("Resume Match Score:", score,"%")

        if score > 60:
            st.success("Good match for this job!")
        else:
            st.warning("You may need to improve your skills.")