from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.db import create_candidate, get_candidate, init_db, list_candidates, update_review
from src.screening import ROLE_SKILLS, VALID_ROLES, extract_pdf_text, score_resume

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset.csv"
ALLOWED_EXTENSIONS = {"pdf", "txt"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
init_db()


def get_dataset_summary() -> dict:
    try:
        frame = pd.read_csv(DATA_PATH)
        return {
            "rows": int(len(frame)),
            "roles": int(frame["job_role"].nunique()),
            "shortlisted": int(frame["shortlisted"].sum()),
            "shortlist_rate": float(frame["shortlisted"].mean()),
            "mean_score": float(frame["predicted_score"].mean()),
        }
    except (OSError, ValueError, KeyError):
        return {"rows": 0, "roles": 0, "shortlisted": 0, "shortlist_rate": 0, "mean_score": 0}


@app.template_filter("pct")
def percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "—"


@app.get("/")
def index():
    records = list_candidates(limit=8)
    return render_template("index.html", summary=get_dataset_summary(), records=records)


@app.route("/screen", methods=["GET", "POST"])
def screen():
    if request.method == "GET":
        return render_template("screen.html", roles=VALID_ROLES, role_skills=ROLE_SKILLS)

    uploaded = request.files.get("resume_file")
    job_description = request.form.get("job_description", "").strip()
    role = request.form.get("job_role", "Software_Engineer")
    if not uploaded or not uploaded.filename:
        flash("Choose a PDF or TXT resume file.", "error")
        return redirect(url_for("screen"))
    safe_name = secure_filename(uploaded.filename)
    extension = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if extension not in ALLOWED_EXTENSIONS:
        flash("Only PDF and TXT files are supported.", "error")
        return redirect(url_for("screen"))
    file_bytes = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        flash("File is larger than the 5 MB demo limit.", "error")
        return redirect(url_for("screen"))
    try:
        if extension == "pdf":
            resume_text = extract_pdf_text(file_bytes)
        else:
            resume_text = file_bytes.decode("utf-8", errors="replace")
        result = score_resume(resume_text, job_description, role)
        # The raw file and its full text remain in memory only, then are discarded.
        candidate_id = create_candidate(result, safe_name)
    except Exception as exc:
        app.logger.exception("Screening failed")
        message = str(exc) if isinstance(exc, ValueError) else "Could not extract this file. Try a text-based PDF or UTF-8 TXT."
        flash(message, "error")
        return redirect(url_for("screen"))
    flash("Screening complete. The recommendation is advisory and awaits human review.", "success")
    return redirect(url_for("candidate_detail", candidate_id=candidate_id))


@app.get("/candidates")
def candidates():
    records = list_candidates(limit=500)
    filter_status = request.args.get("status", "all")
    if filter_status == "pending":
        records = [r for r in records if r["review_status"] == "Pending human review"]
    elif filter_status == "reviewed":
        records = [r for r in records if r["review_status"] == "Reviewed"]
    return render_template("candidates.html", records=records, filter_status=filter_status)


@app.route("/candidate/<int:candidate_id>", methods=["GET", "POST"])
def candidate_detail(candidate_id: int):
    if request.method == "POST":
        decision = request.form.get("decision", "")
        notes = request.form.get("reviewer_notes", "").strip()
        try:
            changed = update_review(candidate_id, decision, notes)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("candidate_detail", candidate_id=candidate_id))
        if not changed:
            flash("Candidate record not found.", "error")
            return redirect(url_for("candidates"))
        flash("Human review decision saved.", "success")
        return redirect(url_for("candidate_detail", candidate_id=candidate_id))
    record = get_candidate(candidate_id)
    if not record:
        flash("Candidate record not found.", "error")
        return redirect(url_for("candidates"))
    return render_template("candidate_detail.html", record=record)


@app.get("/fairness")
def fairness():
    return render_template("fairness.html", summary=get_dataset_summary())


@app.get("/export.csv")
def export_csv():
    records = list_candidates(limit=10000)
    buffer = io.StringIO()
    columns = ["id", "created_at", "resume_label", "job_role", "years_exp", "skills_matched", "required_skill_count", "keyword_density", "similarity_score", "predicted_score", "shortlist_recommendation", "matched_skills", "missing_skills", "review_status", "reviewer_decision", "reviewer_notes"]
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        row = dict(record)
        row["matched_skills"] = "; ".join(row["matched_skills"])
        row["missing_skills"] = "; ".join(row["missing_skills"])
        row["shortlist_recommendation"] = "Yes" if row["shortlist_recommendation"] else "No"
        writer.writerow(row)
    return Response(buffer.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=candidate_reviews.csv"})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "application": "AI Resume Screening Demo", "dataset_rows": get_dataset_summary()["rows"]})


@app.post("/api/score")
def api_score():
    """Stateless JSON endpoint. Does not persist request text or results."""
    payload = request.get_json(silent=True) or {}
    try:
        result = score_resume(str(payload.get("resume_text", "")), str(payload.get("job_description", "")), str(payload.get("job_role", "Software_Engineer")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.errorhandler(413)
def too_large(_error):
    flash("Upload too large. The maximum allowed size is 5 MB.", "error")
    return redirect(url_for("screen"))


if __name__ == "__main__":
    # Development server only. Use a production WSGI server for deployment.
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
