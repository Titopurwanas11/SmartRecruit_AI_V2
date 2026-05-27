"""Radar chart utilities for candidate comparison.

Implements:
- dimension extraction from job description (keyword heuristic)
- per-candidate per-dimension scoring (keyword or embedding-based)
- normalization to 0-100
- building a Plotly radar chart figure

This module intentionally does NOT generate any candidate scores itself — it
provides the functions you can call with real candidate texts / embeddings.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Iterable
import math
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Simple mapping of example dimensions to indicative keywords.
# You can extend or replace this mapping with a project-specific one.
DEFAULT_DIMENSION_KEYWORDS: Dict[str, List[str]] = {
    # General / Soft Skills
    "Communication": ["presentation", "communication", "stakeholder", "collaboration", "team", "verbal", "written", "negotiation", "komunikasi", "presentasi", "kolaborasi", "tim", "lisan", "tulisan", "negosiasi"],
    "Experience": ["years", "experience", "senior", "lead", "manager", "proven", "track record", "pengalaman", "tahun", "manajer", "rekam jejak"],
    "Leadership": ["leadership", "mentoring", "management", "strategy", "agile", "scrum", "coach", "vision", "kepemimpinan", "manajemen", "strategi", "pelatih", "visi", "membimbing"],
    "Problem Solving": ["analytical", "problem solving", "algorithm", "optimization", "critical thinking", "troubleshoot", "decision making", "analitis", "pemecahan masalah", "algoritma", "optimasi", "berpikir kritis", "pengambilan keputusan"],
    "Project Management": ["project management", "pmp", "agile", "scrum", "timeline", "budgeting", "planning", "coordination", "execution", "manajemen proyek", "anggaran", "perencanaan", "koordinasi", "eksekusi"],
    
    # Business / Non-IT Domains
    "Marketing/Sales": ["marketing", "sales", "seo", "campaign", "social media", "b2b", "b2c", "advertising", "conversion", "lead generation", "pemasaran", "penjualan", "kampanye", "media sosial", "periklanan"],
    "Finance/Accounting": ["finance", "accounting", "budget", "financial", "audit", "tax", "payroll", "excel", "reporting", "forecasting", "keuangan", "akuntansi", "anggaran", "pajak", "penggajian", "pelaporan", "peramalan"],
    "HR/Recruitment": ["hr", "human resources", "recruitment", "talent", "onboarding", "interview", "employee", "training", "culture", "sumber daya manusia", "rekrutmen", "wawancara", "karyawan", "pelatihan", "budaya"],
    "Operations/Logistics": ["operations", "logistics", "supply chain", "inventory", "procurement", "vendor", "process", "efficiency", "operasi", "operasional", "logistik", "rantai pasok", "inventaris", "pengadaan", "proses", "efisiensi"],
    "Customer Service": ["customer service", "support", "client", "satisfaction", "helpdesk", "resolution", "retention", "care", "layanan pelanggan", "dukungan", "klien", "kepuasan", "resolusi"],
    "Design/Creative": ["design", "creative", "adobe", "photoshop", "illustrator", "figma", "video", "content", "graphic", "visual", "desain", "kreatif", "konten", "grafis"],
    "Writing/Editing": ["writing", "editing", "copywriting", "content", "proofreading", "journalism", "blog", "articles", "press", "menulis", "penulisan", "mengedit", "konten", "jurnalisme", "artikel"],

    # IT / Technical Domains
    "Programming": ["python", "java", "c++", "golang", "programming", "development", "software", "coding", "pemrograman", "pengembangan", "perangkat lunak"],
    "Database": ["sql", "postgres", "mysql", "mongodb", "database", "bigquery", "nosql", "data modeling", "basis data", "pemodelan data"],
    "Cloud/DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "deployment", "infrastructure", "infrastruktur", "penyebaran"],
    "Data/AI": ["machine learning", "ai", "data science", "statistics", "pandas", "pytorch", "tensorflow", "analytics", "modeling", "sains data", "statistik", "analitik", "pemodelan"],
    "Frontend": ["react", "angular", "vue", "javascript", "html", "css", "ui", "ux", "frontend", "web", "antarmuka"],
    "Backend/API": ["api", "rest", "graphql", "microservices", "node", "django", "spring", "backend", "server", "layanan mikro", "api", "backend"]
}


def extract_dimensions_from_jd(jd_text: str, n_dims: int = 5, mapping: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Pick the top `n_dims` dimensions for a job description using keyword counts.

    Args:
        jd_text: raw job description text.
        n_dims: number of dimensions to return (4-6 recommended).
        mapping: optional dict mapping dimension->keywords to use instead of default.

    Returns:
        Ordered list of dimension names.
    """
    if mapping is None:
        mapping = DEFAULT_DIMENSION_KEYWORDS

    jd_lower = (jd_text or "").lower()
    scores: Dict[str, int] = {}
    for dim, keywords in mapping.items():
        count = 0
        for kw in keywords:
            count += jd_lower.count(kw)
        scores[dim] = count

    # Sort dimensions by keyword hit counts (desc) and return top n_dims
    sorted_dims = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = [dim for dim, _ in sorted_dims[:n_dims]]

    # Fallback: if mapping keywords don't match, return first n_dims of mapping
    if not any(v for _, v in sorted_dims):
        selected = list(mapping.keys())[:n_dims]

    return selected


def _keyword_score_for_dimension(text: str, keywords: Iterable[str]) -> float:
    """Simple keyword-based score (counts occurrences)."""
    t = (text or "").lower()
    s = float(sum(t.count(k) for k in keywords))
    return s


def score_candidate_dimensions(
    candidate_text: str,
    dimensions: List[str],
    mapping: Optional[Dict[str, List[str]]] = None,
    *,
    embedding_model: Optional[object] = None,
) -> Dict[str, float]:
    """Compute raw scores for a candidate across provided dimensions.

    Modes:
      - If `embedding_model` is None => keyword-count heuristic using `mapping`.
      - If `embedding_model` is provided => embedding cosine similarity between
        candidate embedding and each dimension's keyword-joined text.

    Returns a dict dimension -> raw score (not normalized).
    """
    if mapping is None:
        mapping = DEFAULT_DIMENSION_KEYWORDS

    scores: Dict[str, float] = {}

    if embedding_model is None:
        for dim in dimensions:
            keywords = mapping.get(dim, [])
            scores[dim] = _keyword_score_for_dimension(candidate_text, keywords)
        return scores

    # Embedding-based scoring path
    try:
        # The embedding model must expose an encode(list[str]) -> np.ndarray interface
        candidate_emb = embedding_model.encode([candidate_text])[0].reshape(1, -1)
    except Exception as e:
        logger.exception("Embedding model encode failed, falling back to keyword scoring: %s", e)
        for dim in dimensions:
            keywords = mapping.get(dim, [])
            scores[dim] = _keyword_score_for_dimension(candidate_text, keywords)
        return scores

    for dim in dimensions:
        kw_text = " ".join(mapping.get(dim, [])) or dim
        try:
            dim_emb = embedding_model.encode([kw_text])[0].reshape(1, -1)
            sim = float(cosine_similarity(candidate_emb, dim_emb)[0][0])
        except Exception:
            sim = 0.0
        scores[dim] = sim

    return scores





def build_radar_dataframe(
    candidates: List[Dict[str, str]],
    dimensions: List[str],
    mapping: Optional[Dict[str, List[str]]] = None,
    embedding_model: Optional[object] = None,
) -> pd.DataFrame:
    """Return a DataFrame with candidate names as index and normalized scores per-dimension.

    Args:
        candidates: list of dicts with keys `name` and `text`.
        dimensions: list of dimension names (labels for the radar chart).
        mapping: optional dimension->keywords mapping.
        embedding_model: optional model to compute embedding similarity.

    Returns:
        pd.DataFrame: index=candidate names, columns=dimensions, values=0-100 ints.
    """
    names = []
    raw_scores_list = []
    
    for cand in candidates:
        name = cand.get("name") or cand.get("candidate") or "candidate"
        text = cand.get("text") or cand.get("resume_text") or ""
        raw = score_candidate_dimensions(text, dimensions, mapping, embedding_model=embedding_model)
        raw_scores_list.append(raw)
        names.append(name)

    df_raw = pd.DataFrame(raw_scores_list, columns=dimensions, index=names)
    
    # Normalization globally across candidates per dimension
    if embedding_model is not None:
        # Embedding returns cosine similarity [-1, 1]. Map to [0, 100] safely.
        df_norm = df_raw.clip(lower=0, upper=1.0) * 100
    else:
        # Keyword scoring: scale by the max count observed in each dimension
        max_vals = df_raw.max()
        # Avoid division by zero
        max_vals = max_vals.replace(0, 1)
        df_norm = (df_raw / max_vals) * 100

    return df_norm.round().astype(int)


def plotly_radar_chart(
    df: pd.DataFrame,
    title: Optional[str] = "Candidate Comparison",
    radial_range: Optional[list[int]] = None,
) -> go.Figure:
    """Create a Plotly radar (polar) chart comparing rows in `df`.

    Args:
        df: DataFrame with index=candidate names and columns=dimension labels (0-100 ints).
        title: optional chart title.
        radial_range: optional radial axis range, e.g. [0, 100] or [0, 150].

    Returns:
        Plotly `go.Figure` object ready to render in Streamlit via `st.plotly_chart(fig)`.
    """
    if df.empty:
        raise ValueError("DataFrame is empty — provide candidate scores before plotting.")

    radial_range = radial_range or [0, 100]
    categories = list(df.columns)
    fig = go.Figure()

    # Use Plotly qualitative palette, convert to rgba for fill
    colors = [
        ("rgba(31, 119, 180, 0.3)", "#1f77b4"),
        ("rgba(255, 127, 14, 0.3)", "#ff7f0e"),
        ("rgba(44, 160, 44, 0.3)", "#2ca02c"),
        ("rgba(214, 39, 40, 0.3)", "#d62728"),
        ("rgba(148, 103, 189, 0.3)", "#9467bd"),
        ("rgba(140, 86, 75, 0.3)", "#8c564b"),
    ]

    for i, (name, row) in enumerate(df.iterrows()):
        values = row.tolist()
        fill_color, line_color = colors[i % len(colors)]
        # Radar charts close the polygon by repeating the first value
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=fill_color,
                name=str(name),
                line=dict(color=line_color, width=2),
                hovertemplate="%{theta}: %{r}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                range=radial_range,
                tickvals=[
                    radial_range[0],
                    (radial_range[0] + radial_range[1]) // 4,
                    (radial_range[0] + radial_range[1]) // 2,
                    (3 * radial_range[1]) // 4,
                    radial_range[1],
                ],
            ),
            angularaxis=dict(direction="clockwise"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
