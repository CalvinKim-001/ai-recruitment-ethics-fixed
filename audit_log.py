"""
audit_log.py
------------
Transparent audit logging system for the AI recruitment dashboard.

WHY AUDIT LOGS MATTER (ETHICALLY):
AI systems in high-stakes decisions must be accountable and traceable.
An audit log answers the question: "What did the AI recommend, when,
why, and what did the human decide to do about it?"

This is not just good practice — it is increasingly a legal requirement:
- EU AI Act (2024): High-risk AI systems must maintain logs of operation
- NYC Local Law 144 (2023): Bias audit results must be documented
- EEOC guidance: Algorithmic hiring decisions must be reviewable

Without an audit trail, AI discrimination is nearly impossible to detect
or challenge after the fact. Transparency starts with logging.

This module stores logs locally in a JSON file — simple, transparent,
and easy for non-technical audiences to inspect.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Default log file location (stored locally, not in any external database)
DEFAULT_LOG_PATH = Path("data/audit_log.json")


def _load_log(log_path: Path = DEFAULT_LOG_PATH) -> list:
    """Load existing audit log or return empty list."""
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_log(entries: list, log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Save audit log to disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_evaluation(
    candidate_name: str,
    candidate_features: dict,
    biased_score: float,
    fair_score: float,
    biased_recommendation: str,
    fair_recommendation: str,
    model_used: str = "Both",
    log_path: Path = DEFAULT_LOG_PATH
) -> str:
    """
    Log a candidate evaluation event.

    Every time the AI evaluates a candidate, this creates a permanent
    record of what it recommended and why (via scores).

    Returns: the evaluation_id for this record
    """
    entries = _load_log(log_path)

    evaluation_id = f"EVAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(entries):04d}"

    entry = {
        "evaluation_id": evaluation_id,
        "timestamp": datetime.now().isoformat(),
        "event_type": "AI_EVALUATION",
        "candidate_name": candidate_name,
        "candidate_features": candidate_features,
        "model_used": model_used,
        "biased_model_score": round(float(biased_score), 4),
        "biased_model_recommendation": biased_recommendation,
        "fair_model_score": round(float(fair_score), 4),
        "fair_model_recommendation": fair_recommendation,
        "human_review_required": True,  # Always True — AI is decision SUPPORT only
        "recruiter_override": None,     # Populated when recruiter acts
        "recruiter_decision": None,
        "recruiter_notes": None,
        "override_timestamp": None,
    }

    entries.append(entry)
    _save_log(entries, log_path)

    return evaluation_id


def log_recruiter_override(
    evaluation_id: str,
    recruiter_decision: str,  # "APPROVED", "REJECTED", "DEFERRED"
    recruiter_notes: str = "",
    log_path: Path = DEFAULT_LOG_PATH
) -> bool:
    """
    Record a recruiter's override of the AI recommendation.

    ETHICAL NOTE:
    Recruiter overrides are one of the most important data points in
    responsible AI governance. They tell us:
    1. How often humans disagree with the AI
    2. Whether override patterns themselves show bias
    3. Whether the AI is calibrated to recruiter judgment

    Over time, override patterns can be used to improve the model —
    this is the "human-in-the-loop learning" process.

    Returns: True if the override was successfully logged
    """
    entries = _load_log(log_path)

    for entry in entries:
        if entry["evaluation_id"] == evaluation_id:
            entry["recruiter_override"] = True
            entry["recruiter_decision"] = recruiter_decision
            entry["recruiter_notes"] = recruiter_notes
            entry["override_timestamp"] = datetime.now().isoformat()
            _save_log(entries, log_path)
            return True

    return False  # evaluation_id not found


def get_recent_evaluations(n: int = 20, log_path: Path = DEFAULT_LOG_PATH) -> list:
    """Return the n most recent evaluation log entries."""
    entries = _load_log(log_path)
    return sorted(entries, key=lambda x: x["timestamp"], reverse=True)[:n]


def get_override_statistics(log_path: Path = DEFAULT_LOG_PATH) -> dict:
    """
    Compute statistics on recruiter overrides.

    Returns aggregated stats useful for governance reporting:
    - Total evaluations
    - Override rate
    - Decision breakdown
    """
    entries = _load_log(log_path)

    if not entries:
        return {
            "total_evaluations": 0,
            "total_overrides": 0,
            "override_rate_pct": 0,
            "approved_overrides": 0,
            "rejected_overrides": 0,
            "deferred_overrides": 0,
        }

    total = len(entries)
    overrides = [e for e in entries if e.get("recruiter_override")]
    approved = [e for e in overrides if e.get("recruiter_decision") == "APPROVED"]
    rejected = [e for e in overrides if e.get("recruiter_decision") == "REJECTED"]
    deferred = [e for e in overrides if e.get("recruiter_decision") == "DEFERRED"]

    return {
        "total_evaluations": total,
        "total_overrides": len(overrides),
        "override_rate_pct": round(len(overrides) / total * 100, 1) if total > 0 else 0,
        "approved_overrides": len(approved),
        "rejected_overrides": len(rejected),
        "deferred_overrides": len(deferred),
    }


def clear_log(log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Clear all audit log entries. Use with caution — for demo resets only."""
    _save_log([], log_path)


def export_log_as_dataframe(log_path: Path = DEFAULT_LOG_PATH):
    """Export audit log as a pandas DataFrame for analysis."""
    import pandas as pd
    entries = _load_log(log_path)
    if not entries:
        return pd.DataFrame()

    rows = []
    for e in entries:
        rows.append({
            "evaluation_id": e.get("evaluation_id"),
            "timestamp": e.get("timestamp"),
            "candidate_name": e.get("candidate_name"),
            "biased_score": e.get("biased_model_score"),
            "fair_score": e.get("fair_model_score"),
            "biased_recommendation": e.get("biased_model_recommendation"),
            "fair_recommendation": e.get("fair_model_recommendation"),
            "recruiter_override": e.get("recruiter_override"),
            "recruiter_decision": e.get("recruiter_decision"),
            "recruiter_notes": e.get("recruiter_notes"),
            "override_timestamp": e.get("override_timestamp"),
        })

    return pd.DataFrame(rows)
