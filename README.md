# Smart Recruit AI v2
> **Intelligent Resume-to-Job Matching System via Semantic Embedding**

Smart Recruit AI v2 is a high-performance recruitment platform that leverages **Natural Language Processing (NLP)** and **Semantic Search** to bridge the gap between job seekers and employers. Unlike traditional keyword-based ATS, our system understands the deep context of professional experiences using **Sentence-BERT (SBERT)**.

## 🚀 Overview

Recruitment is broken. Thousands of qualified candidates are often lost in the "Resume Black Hole" because traditional systems filter based on exact keyword matches. Smart Recruit AI v2 solves this by:
- **Semantic Understanding:** Representing competence in a 384-dimensional vector space.
- **High Accuracy:** Achieving **97.75% accuracy** in job role classification.
- **Dual Operational Modes:** Optimized workflows for both Job Seekers and HR Professionals.

## ✨ Key Features

### 1. Job Seeker Mode
- **Resume Parsing:** Automatic text extraction from PDF resumes.
- **Role Prediction:** Predicts your ideal job category among 20+ professional fields.
- **Semantic Matching:** Matches your profile against a curated database of 20,000+ LinkedIn job postings.
- **Skill Gap Analysis:** Identifies matched skills and suggests missing competencies.

### 2. HR Mode (Batch Processing)
- **Bulk Upload:** Upload multiple CVs in PDF or ZIP format.
- **Automated Ranking:** Ranks candidates based on semantic relevance to a specific job description.
- **Efficiency:** Processes hundreds of resumes in seconds without requiring GPU acceleration.

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **NLP & ML:** `sentence-transformers` (SBERT), `scikit-learn` (Logistic Regression + TF-IDF), `nltk`
- **UI Framework:** `streamlit`
- **PDF Engine:** `PyMuPDF` (Fitz)
- **Data Handling:** `pandas`, `numpy`

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amirullahh/SmartRecruit_AI_V2.git
   cd SmartRecruit_AI_V2
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Data & Model Setup

To keep the repository lightweight, datasets and pre-computed models are not included. Follow these steps to set up:

1. **Acquire Data:** Download the [LinkedIn Job Postings 2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) and [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) from Kaggle and place them in `data/raw/`.
2. **Train Classifier:**
   ```bash
   python src/models/train_classifier.py
   ```
3. **Generate Embeddings:** Run the embedding script to process job descriptions.
   ```bash
   python scripts/generate_embeddings.py
   ```

## 🖥️ Usage

Run the Streamlit dashboard:
```bash
streamlit run ui/app.py
```

## 📂 Project Structure

```text
├── scripts/             # Utility scripts for data generation
├── src/                 # Core source code
│   ├── features/        # Feature engineering (TF-IDF)
│   ├── matching/        # Semantic similarity logic
│   ├── models/          # Model training and inference
│   ├── nlp/             # Skill extraction and NLP tasks
│   ├── pipeline/        # Integrated operational pipelines
│   └── preprocessing/   # Text cleaning and normalization
├── tests/               # Comprehensive unit test suite
└── ui/                  # Streamlit frontend implementation
```

## 👥 Contributors
- **Amirullah Hidayat** - Project Lead & ML Engineer
- **Farhan Alwanda** - NLP Engineer
- **Muhammad Trio Novrian** - Embedding Engineer
- **Yohanes Aditya Krismawan** - Backend & Features Engineer
- **Tito Purwana S** - Frontend Engineer

---
*Developed in collaboration with IBM Skillsbuild.*
