import streamlit as st
import fitz  # PyMuPDF
import sys
import os

# Ensure project root is in sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.pipeline import run_pipeline

# 1. Layout & Branding
st.set_page_config(page_title="Smart Recruit AI", layout="wide")
st.title("Smart Recruit AI")
st.caption("AI-powered resume analysis and job matching")

# 2. Sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Input resume text or upload PDF.
    2. AI cleans and processes the text.
    3. Classifier predicts your most suitable job role.
    4. Matcher finds the best job listings and calculates skill gaps.
    """)

# 3. Input Tabs
tab1, tab2, tab3 = st.tabs(["Resume Text", "Upload PDF", "HR Mode"])

resume_text = ""

with tab1:
    input_text = st.text_area("Paste your resume text here:", height=350, key="resume_input")
    if st.button("Analyze Resume"):
        if input_text.strip():
            resume_text = input_text
        else:
            st.warning("Please paste some text first.")

with tab2:
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # 4. PDF Extraction
        try:
            # We use a button for consistency or just trigger on upload
            # Requirement didn't specify a button for Tab 2, but it's often better.
            # However, I'll extract text immediately upon upload to make it available.
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            if text.strip():
                resume_text = text
            else:
                st.error("Could not extract text from the uploaded PDF.")
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")

with tab3:
    st.subheader("HR Mode — Rank Candidates Against a Job Description")

    jd_input = st.text_area(
        "Paste the Job Description here:",
        height=250,
        key="jd_input"
    )

    st.markdown("**Upload Resumes** (multiple PDFs, or a single ZIP containing PDFs):")
    uploaded_resumes = st.file_uploader(
        "Choose PDF(s) or a ZIP file",
        type=["pdf", "zip"],
        accept_multiple_files=True,
        key="hr_upload"
    )

    hr_clicked = st.button("Rank Candidates")
    if hr_clicked:
        if not jd_input.strip():
            st.warning("Please paste a job description before ranking.")
        elif not uploaded_resumes:
            st.info("Upload at least one resume PDF (or a ZIP) to start ranking.")
        else:
            from src.pipeline.hr_pipeline import run_hr_pipeline
            from src.utils.resume_loader import extract_text_from_pdf, extract_resumes_from_zip, extract_owner_name
            import pandas as pd

            resumes = []
            for f in uploaded_resumes:
                raw = f.read()
                if f.name.lower().endswith(".zip"):
                    resumes.extend(extract_resumes_from_zip(raw))
                else:
                    pdf_text = extract_text_from_pdf(raw, f.name)
                    candidate_filename = f.name.rsplit(".", 1)[0]
                    owner_name = extract_owner_name(pdf_text)
                    resumes.append((candidate_filename, owner_name, pdf_text))

            if not resumes:
                st.error("No readable PDFs found in the uploaded files.")
            else:
                with st.spinner(f"Scoring {len(resumes)} candidate(s)..."):
                    hr_results = run_hr_pipeline(jd_input, resumes)

                st.success(f"Ranked {len(hr_results)} candidate(s)")
                df = pd.DataFrame(hr_results)[["rank", "candidate", "owner_name", "score"]]
                df.columns = ["Rank", "Candidate File", "Owner Name", "Similarity Score"]
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.caption("Score ranges from 0.0 (no match) to 1.0 (perfect match). Recommended threshold: ≥ 0.50")

# 5. Pipeline Integration
if resume_text:
    try:
        with st.spinner("Analyzing your resume..."):
            results = run_pipeline(resume_text)

        # 6. Results Display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted Role")
            st.success(results["predicted_role"])

            st.subheader("Your Skills")
            st.write(", ".join(results["extracted_skills"]) or "No skills detected")

        with col2:
            st.subheader("Skill Gap (vs. Top Match)")
            if results["top_matches"]:
                matched = ", ".join(results["skill_gap"]["matched_skills"]) or "None"
                missing = ", ".join(results["skill_gap"]["missing_skills"]) or "None"
                st.markdown(f"**Matched:** {matched}")
                st.markdown(f"**Missing:** {missing}")
            else:
                st.info("No sufficiently similar jobs found to compute skill gap.")

        st.subheader("Top Job Matches")
        if results["top_matches"]:
            for i, match in enumerate(results["top_matches"]):
                with st.expander(f"{i+1}. {match['title']} at {match['company']} (Score: {match['score']:.2f})"):
                    st.write(f"**Required Skills:** {', '.join(match['required_skills'])}")
        else:
            st.warning("No job matches found above the similarity threshold (0.50). Try a more detailed resume.")

        # 7. AI Advisor — on-demand only
        if results["top_matches"]:
            st.divider()
            st.subheader("AI Resume Advisor")
            st.caption("Get specific suggestions from Gemini on how to improve your resume for your top match.")

            if st.button("Get AI Suggestions", key="ai_advisor_btn"):
                st.session_state["ai_advice"] = None
                from src.llm.gemini_advisor import generate_resume_advice
                try:
                    with st.spinner("Asking Gemini for advice..."):
                        advice = generate_resume_advice(
                            resume_text,
                            results["top_matches"][0],
                            results["skill_gap"],
                        )
                    st.session_state["ai_advice"] = advice
                except RuntimeError as e:
                    st.error(str(e))

            if st.session_state.get("ai_advice"):
                st.markdown(st.session_state["ai_advice"])

    except Exception as e:
        # 8. Error Handling
        st.error(str(e))
else:
    if not resume_text:
        st.info("Please paste or upload a resume to begin.")
