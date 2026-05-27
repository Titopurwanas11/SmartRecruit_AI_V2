import streamlit as st
import fitz  # PyMuPDF
import sys
import os
import pandas as pd

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
            from src.utils.resume_loader import extract_text_from_pdf, extract_resumes_from_zip, extract_owner_name_from_pdf
            import pandas as pd

            resumes = []
            for f in uploaded_resumes:
                raw = f.read()
                if f.name.lower().endswith(".zip"):
                    resumes.extend(extract_resumes_from_zip(raw))
                else:
                    pdf_text = extract_text_from_pdf(raw, f.name)
                    candidate_filename = f.name.rsplit(".", 1)[0]
                    owner_name = extract_owner_name_from_pdf(raw, f.name)
                    resumes.append((candidate_filename, owner_name, pdf_text))

            if not resumes:
                st.error("No readable PDFs found in the uploaded files.")
            else:
                with st.spinner(f"Scoring {len(resumes)} candidate(s)..."):
                    hr_results = run_hr_pipeline(jd_input, resumes)

                st.session_state["hr_results"] = hr_results
                st.session_state["hr_resumes"] = resumes
                st.session_state["interview_questions"] = None

    if st.session_state.get("hr_results"):
        hr_results = st.session_state["hr_results"]
        resumes = st.session_state["hr_resumes"]

        st.success(f"Ranked {len(hr_results)} candidate(s)")
        df = pd.DataFrame(hr_results)[["rank", "candidate", "owner_name", "score", "contact"]]
        df.columns = ["Rank", "Candidate File", "Owner Name", "Similarity Score", "Contact"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        if hr_results:
            candidate_choices = [
                f"{r['rank']}. {r['candidate']} ({r['owner_name']})"
                for r in hr_results
            ]
            selected_index = st.selectbox(
                "Select candidate for interview questions",
                options=list(range(len(candidate_choices))),
                format_func=lambda i: candidate_choices[i],
                key="selected_candidate_index",
            )

            with st.expander("Interview Question Generator"):
                st.write(
                    "Generate personalized technical and behavioral questions for the selected candidate "
                    "based on the job description and resume content."
                )

                if st.button("Generate Interview Questions", key="generate_interview_questions_btn"):
                    from src.llm.interview_question_generator import generate_interview_questions
                    from src.nlp.skill_extraction import load_skill_list, extract_skills
                    from src.preprocessing.clean_text import clean_text
                    import config

                    selected_candidate = hr_results[selected_index]
                    selected_resume = next(
                        (
                            raw_text
                            for file_name, _, raw_text in resumes
                            if file_name == selected_candidate["candidate"]
                        ),
                        ""
                    )

                    if not selected_resume:
                        st.error("Unable to load the selected candidate's resume text.")
                    else:
                        cleaned_resume = clean_text(selected_resume)
                        cleaned_jd = clean_text(jd_input)
                        skill_list = load_skill_list(config.SKILL_LIST_PATH)
                        resume_skills = extract_skills(cleaned_resume, skill_list)
                        jd_skills = extract_skills(cleaned_jd, skill_list)

                        matched_skills = sorted(set(resume_skills).intersection(jd_skills))
                        missing_skills = sorted(set(jd_skills).difference(resume_skills))
                        target_role = "Target role from the job description"

                        st.session_state["interview_questions"] = None
                        try:
                            with st.spinner("Generating interview questions..."):
                                st.session_state["interview_questions"] = generate_interview_questions(
                                    target_role=target_role,
                                    job_description=jd_input,
                                    matched_skills=matched_skills,
                                    missing_skills=missing_skills,
                                    resume_text=selected_resume,
                                )
                        except RuntimeError as e:
                            st.error(str(e))

                if st.session_state.get("interview_questions"):
                    st.subheader("Generated Interview Questions")
                    st.markdown(st.session_state["interview_questions"])

            # Candidate Comparison Radar Chart
            with st.expander("Compare Candidates (Radar Chart)"):
                st.write("Visualize top candidates across selected skill dimensions.")

                if len(hr_results) < 2:
                    st.info("Need at least 2 ranked candidates to compare. Upload more resumes and re-run ranking.")
                else:
                    try:
                        from src.visualization.radar import (
                            extract_dimensions_from_jd,
                            build_radar_dataframe,
                            plotly_radar_chart,
                        )
                        from src.matching.embeddings import load_sbert_model
                    except Exception as e:
                        st.error(f"Radar chart import error: {e}")
                        st.warning('Install required packages: pip install plotly')
                    else:
                        # Candidate selection dropdown
                        # Use real name for label, fallback to filename if needed
                        candidate_options = [f"{r['rank']}. {r['owner_name'] if r['owner_name'] != 'Unknown' else r['candidate']}" for r in hr_results]
                        # Default: only top-1 candidate selected
                        selected = st.multiselect("Select candidates", options=candidate_options, default=[candidate_options[0]])

                        # Dimension selection dropdown
                        all_dims = extract_dimensions_from_jd(jd_input, n_dims=99)
                        default_dims = all_dims[:8] if len(all_dims) >= 8 else all_dims
                        dimensions = st.multiselect("Select dimensions", options=all_dims, default=default_dims)

                        scoring_mode = st.selectbox("Scoring mode", options=["keyword", "embedding"], index=0)

                        # Prepare candidate list
                        selected_candidates = []
                        for sel in selected:
                            # sel format: "{rank}. {owner_name}" (or fallback to filename)
                            label = sel.split('. ', 1)[1] if '. ' in sel else sel
                            # Find resume by owner_name first, fallback to filename
                            match = next((r for r in hr_results if (r['owner_name'] != 'Unknown' and r['owner_name'] == label) or r['candidate'] == label), None)
                            if match:
                                raw_text = next((raw for file, _, raw in resumes if file == match['candidate']), "")
                                selected_candidates.append({"name": match['owner_name'] if match['owner_name'] != 'Unknown' else match['candidate'], "text": raw_text})
                            else:
                                st.warning(f"Resume text not found for {sel}")

                        if selected_candidates and dimensions:

                            embedding_model = None
                            radar_range = [0, 100]
                            if scoring_mode == "embedding":
                                try:
                                    embedding_model = load_sbert_model()
                                    radar_range = [0, 100]  # Perbesar jika ingin, misal [0, 120] atau [0, 150]
                                except Exception as e:
                                    st.warning(f"Embedding model load failed, falling back to keyword mode: {e}")
                                    embedding_model = None
                            try:
                                df_radar = build_radar_dataframe(selected_candidates, dimensions, embedding_model=embedding_model)
                                from src.visualization import radar as radar_mod
                                fig = radar_mod.plotly_radar_chart(df_radar, title="Top candidates", radial_range=radar_range)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Failed to build radar chart: {e}")

            if st.session_state.get("ai_advice"):
                st.markdown(st.session_state["ai_advice"])
        else:
            if not resume_text:
                st.info("Please paste or upload a resume to begin.")
