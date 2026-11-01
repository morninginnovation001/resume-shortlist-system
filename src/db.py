"""SQLite persistence for demo review records; raw resumes are never stored."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(__file__).resolve().parents[1] / "instance" / "screening.db"


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or os.environ.get("SCREENING_DB_PATH", DEFAULT_DB))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                resume_label TEXT NOT NULL,
                job_role TEXT NOT NULL,
                years_exp INTEGER,
                skills_matched INTEGER NOT NULL,
                required_skill_count INTEGER NOT NULL,
                keyword_density REAL NOT NULL,
                similarity_score REAL NOT NULL,
                predicted_score REAL NOT NULL,
                shortlist_recommendation INTEGER NOT NULL,
                matched_skills_json TEXT NOT NULL,
                missing_skills_json TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Pending human review',
                reviewer_decision TEXT,
                reviewer_notes TEXT
            )
        """)
        conn.commit()


def create_candidate(result: dict[str, Any], resume_label: str, db_path: str | Path | None = None) -> int:
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(db_path) as conn:
        cur = conn.execute("""
            INSERT INTO candidates (
                created_at, resume_label, job_role, years_exp, skills_matched,
                required_skill_count, keyword_density, similarity_score,
                predicted_score, shortlist_recommendation, matched_skills_json,
                missing_skills_json, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, resume_label, result["job_role"], result.get("years_exp"),
            result["skills_matched"], result["required_skill_count"],
            result["keyword_density"], result["similarity_score"],
            result["predicted_score"], int(result["shortlist_recommendation"]),
            json.dumps(result["matched_skills"]), json.dumps(result["missing_skills"]),
            "Pending human review",
        ))
        conn.commit()
        return int(cur.lastrowid)


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    item["matched_skills"] = json.loads(item.pop("matched_skills_json"))
    item["missing_skills"] = json.loads(item.pop("missing_skills_json"))
    item["shortlist_recommendation"] = bool(item["shortlist_recommendation"])
    return item


def get_candidate(candidate_id: int, db_path: str | Path | None = None) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    return _dict(row)


def list_candidates(limit: int = 500, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM candidates ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [_dict(row) for row in rows]


def update_review(candidate_id: int, decision: str, notes: str, db_path: str | Path | None = None) -> bool:
    if decision not in {"Shortlist", "Do not shortlist", "Needs more information"}:
        raise ValueError("Unsupported review decision")
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute("""
            UPDATE candidates
               SET reviewer_decision = ?, reviewer_notes = ?, review_status = 'Reviewed'
             WHERE id = ?
        """, (decision, (notes or "")[:2000], candidate_id))
        conn.commit()
        return cur.rowcount > 0
