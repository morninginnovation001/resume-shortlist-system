# Ethics, privacy, and limitations

## Scope

This is a student demonstration of resume-to-job-description text matching. It is not a validated decision system, and it must not be used to automatically hire, reject, rank, or otherwise make consequential decisions about real people.

## Fairness audit status

The supplied dataset contains feature vectors and a synthetic binary label, but no protected-group attributes. Therefore, this project cannot honestly report demographic group metrics, such as selection-rate ratios or group-wise false-positive / false-negative rates. Do not infer protected attributes from names or resume language, and do not create synthetic protected-group labels to imply a real fairness audit. A proper audit needs an appropriate, lawfully collected, quality-reviewed dataset, clear governance, and qualified interpretation.

## Privacy safeguards in this prototype

- The resume file is handled in memory and is not saved by the web application.
- The full extracted resume text is not written to SQLite.
- Only summary screening signals, detected skill terms, filename label, and reviewer decision/rationale are persisted.
- Filenames and notes may still contain personal data; use fictional examples only in this academic repository.

## Known technical limitations

- TF-IDF uses token overlap and does not understand the meaning of a resume.
- Skill matching is based on exact terms in an editable taxonomy and can miss synonyms, context, spelling variants, and transferable skills.
- PDF text extraction may fail for scanned/image-only or unusually formatted resumes.
- The 65/35 score weights and 65-point review threshold are illustrative, unvalidated parameters.
- Experience extraction is best-effort, informational only, and is not used in the score.
- Synthetic dataset metrics do not prove generalization to real candidates.
- No demographic fairness measurement is possible using the included columns.

## Minimum safeguards before a real pilot

1. Complete an authorized privacy and security assessment, including lawful basis, consent/notice, retention and deletion.
2. Validate every scoring criterion with relevant job experts and document why it is job-related.
3. Test performance, accessibility, parsing failures and disparate impact on appropriate, representative data.
4. Keep a meaningful reviewer decision and an avenue for reconsideration; no automatic rejection by score.
5. Version the code, dataset, criteria, weights and threshold; log only necessary information and audit access.
6. Obtain organizational and legal approval before processing real applicant data.

## Synthetic data statement

The feature dataset contains synthetic records. `predicted_score` and `shortlisted` are generated/simulated fields, not externally verified hiring outcomes. The baseline script is included for demonstrating ML workflow only; it is not used to score uploaded resumes.
