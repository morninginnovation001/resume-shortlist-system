"""Explainable, local-only resume-to-job-description matching utilities.

The live ranking is intentionally a transparent heuristic + TF-IDF similarity,
not the separate classifier trained on the synthetic feature dataset.
"""
from __future__ import annotations

import io
import re
from typing import Any

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROLE_SKILLS: dict[str, list[str]] = {
    "Data_Analyst": ["SQL", "Python", "Excel", "Power BI", "Tableau", "statistics", "pandas", "data cleaning", "data visualization", "ETL", "A/B testing", "Looker"],
    "Software_Engineer": ["Python", "Java", "C++", "JavaScript", "TypeScript", "data structures", "algorithms", "REST API", "Git", "SQL", "Docker", "testing", "React", "Node.js"],
    "HR_Executive": ["recruitment", "talent acquisition", "HRIS", "employee relations", "onboarding", "payroll", "performance management", "workforce planning", "Excel", "communication"],
    "QA_Engineer": ["test automation", "Selenium", "Playwright", "pytest", "JUnit", "API testing", "Postman", "regression testing", "bug tracking", "SQL", "CI/CD", "performance testing"],
    "Project_Manager": ["project planning", "Agile", "Scrum", "Jira", "risk management", "stakeholder management", "budgeting", "resource planning", "communication", "roadmap", "delivery"],
    "ML_Engineer": ["Python", "scikit-learn", "TensorFlow", "PyTorch", "machine learning", "deep learning", "pandas", "NumPy", "feature engineering", "model evaluation", "MLOps", "SQL", "statistics"],
    "DevOps_Engineer": ["Linux", "Docker", "Kubernetes", "AWS", "Azure", "CI/CD", "Terraform", "Ansible", "Bash", "Python", "monitoring", "Git", "networking"],
    "Cybersecurity_Analyst": ["network security", "SIEM", "incident response", "vulnerability assessment", "penetration testing", "Linux", "firewalls", "IAM", "threat analysis", "Python", "NIST", "log analysis"],
    "Business_Analyst": ["requirements gathering", "SQL", "Excel", "Power BI", "process mapping", "stakeholder management", "user stories", "data analysis", "documentation", "UAT", "Agile"],
    "UI_UX_Designer": ["Figma", "wireframing", "prototyping", "user research", "usability testing", "design systems", "accessibility", "information architecture", "interaction design", "visual design", "Adobe XD"],
}

# Small cross-role vocabulary for detecting specific requirements in pasted JDs.
ALL_SKILLS = sorted({s for values in ROLE_SKILLS.values() for s in values}, key=len, reverse=True)
VALID_ROLES = sorted(ROLE_SKILLS.keys())


def _contains_term(text: str, term: str) -> bool:
    """Case-insensitive phrase/term match with safe boundaries for symbols."""
    text_l = text.lower()
    term_l = term.lower()
    # For technical terms containing punctuation (C++, Node.js, CI/CD), exact
    # substring matching avoids the limitations of \b around punctuation.
    if any(ch in term for ch in "+#./-"):
        return term_l in text_l
    return re.search(r"(?<![a-z0-9])" + re.escape(term_l) + r"(?![a-z0-9])", text_l) is not None


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF in memory. The uploaded file is not saved."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = []
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_years_experience(text: str) -> int | None:
    """Best-effort extraction of an explicit years-of-experience phrase."""
    patterns = [
        r"\b(\d{1,2})\s*\+?\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience\b",
        r"\bexperience\s*[:\-]\s*(\d{1,2})\s*\+?\s*years?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(int(match.group(1)), 40)
    return None


def _select_required_skills(job_description: str, role: str) -> list[str]:
    """Prefer known skill terms actually present in the JD; otherwise use role defaults."""
    detected = [skill for skill in ALL_SKILLS if _contains_term(job_description, skill)]
    if len(detected) >= 3:
        # Preserve stable taxonomy ordering and remove any duplicate aliases.
        return detected[:20]
    return ROLE_SKILLS.get(role, ROLE_SKILLS["Software_Engineer"])


def score_resume(resume_text: str, job_description: str, role: str) -> dict[str, Any]:
    """Return an explainable recommendation for human review.

    Components: 65% exact taxonomy skill coverage and 35% TF-IDF cosine
    similarity. A cosine score is lexical similarity, not semantic understanding.
    The 65-point/35-point split is a demo setting and must be calibrated before
    any real hiring use. No protected attributes are inferred or used.
    """
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()
    if len(resume_text) < 30:
        raise ValueError("The extracted resume text is too short. Try a text-based PDF or TXT file.")
    if len(job_description) < 20:
        raise ValueError("Please provide a fuller job description (at least 20 characters).")
    if role not in ROLE_SKILLS:
        role = "Software_Engineer"

    required_skills = _select_required_skills(job_description, role)
    matched_skills = [skill for skill in required_skills if _contains_term(resume_text, skill)]
    missing_skills = [skill for skill in required_skills if skill not in matched_skills]
    coverage = len(matched_skills) / max(len(required_skills), 1)

    # This is a standard lexical TF-IDF representation; it does not infer meaning.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
    matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    skill_component = 65.0 * coverage
    text_component = 35.0 * similarity
    score = round(max(0.0, min(100.0, skill_component + text_component)), 1)
    years = extract_years_experience(resume_text)
    return {
        "job_role": role,
        "years_exp": years,
        "skills_matched": len(matched_skills),
        "required_skill_count": len(required_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "keyword_density": round(coverage, 4),
        "similarity_score": round(similarity, 4),
        "skill_component": round(skill_component, 2),
        "text_component": round(text_component, 2),
        "predicted_score": score,
        "shortlist_recommendation": score >= 65.0,
        "review_status": "Pending human review",
    }
