import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.db import create_candidate, get_candidate, init_db, update_review
from src.screening import ROLE_SKILLS, score_resume

ROOT = Path(__file__).resolve().parents[1]


class ScreeningTests(unittest.TestCase):
    def test_scoring_returns_explainable_fields(self):
        resume = ("Software engineer with 4 years of experience building Python and JavaScript services. "
                  "Created REST API endpoints, used Git, Docker, SQL, automated testing and data structures.")
        jd = "We need a software engineer with Python, REST API, Git, Docker, SQL, testing and data structures experience."
        result = score_resume(resume, jd, "Software_Engineer")
        self.assertGreaterEqual(result["predicted_score"], 0)
        self.assertLessEqual(result["predicted_score"], 100)
        self.assertIn("Python", result["matched_skills"])
        self.assertIn("missing_skills", result)
        self.assertEqual(result["years_exp"], 4)
        self.assertIn(result["review_status"], {"Pending human review"})

    def test_short_text_is_rejected(self):
        with self.assertRaises(ValueError):
            score_resume("too short", "A sufficiently long job description for testing.", "Software_Engineer")

    def test_unknown_role_falls_back_safely(self):
        result = score_resume("Experienced programmer using Python, Git and APIs in production software teams.",
                              "Python, Git, APIs and testing are required for this position.", "Unknown_Role")
        self.assertEqual(result["job_role"], "Software_Engineer")

    def test_dataset_schema(self):
        df = pd.read_csv(ROOT / "data" / "dataset.csv")
        self.assertEqual(len(df), 5000)
        self.assertEqual(len(df.columns), 8)
        self.assertTrue(df["resume_id"].is_unique)
        self.assertEqual(df.isna().sum().sum(), 0)
        self.assertEqual(df["job_role"].nunique(), 10)
        self.assertTrue(set(df["shortlisted"].unique()).issubset({0, 1}))

    def test_human_review_persists_in_temp_db(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "test.db"
            init_db(db_path)
            payload = {
                "job_role": "Software_Engineer", "years_exp": 4, "skills_matched": 2,
                "required_skill_count": 3, "keyword_density": .66, "similarity_score": .55,
                "predicted_score": 58.2, "shortlist_recommendation": False,
                "matched_skills": ["Python", "Git"], "missing_skills": ["SQL"],
            }
            candidate_id = create_candidate(payload, "sample_resume.txt", db_path)
            self.assertTrue(update_review(candidate_id, "Shortlist", "Relevant project evidence.", db_path))
            saved = get_candidate(candidate_id, db_path)
            self.assertEqual(saved["reviewer_decision"], "Shortlist")
            self.assertEqual(saved["review_status"], "Reviewed")


if __name__ == "__main__":
    unittest.main()
