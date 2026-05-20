"""
fairness_audit.py
-----------------
Calculates and interprets fairness metrics for both recruitment models.

METRICS COVERED:
1. Demographic Parity         → Equality of outcome (egalitarian ethics)
2. Equal Opportunity          → Meritocratic fairness
3. Equalized Odds             → Procedural fairness
4. Disparate Impact Ratio     → Anti-discrimination law standard
5. Confusion Matrices by Gender → Error rate comparison

ETHICAL NOTE:
Fairness is not a single number. Different metrics reflect different
ethical theories, and they can mathematically CONFLICT with each other.
Maximizing one fairness measure may reduce another.
This is not a flaw in the tools — it reflects genuine moral complexity.
Reasonable people can disagree about which fairness definition matters most.

This module uses Fairlearn (Microsoft Research) for standardized metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_full_audit(y_true, y_pred_biased, y_pred_fair, gender, model_names=None):
    """
    Run a complete fairness audit comparing biased and fair models.

    Parameters:
        y_true: actual hiring outcomes (ground truth)
        y_pred_biased: predictions from biased baseline model
        y_pred_fair: predictions from fairness-aware model
        gender: sensitive attribute (gender labels)

    Returns:
        dict containing all fairness metrics for both models
    """
    if model_names is None:
        model_names = ["Biased Baseline", "Fairness-Aware"]

    results = {}

    for name, y_pred in zip(model_names, [y_pred_biased, y_pred_fair]):
        results[name] = compute_fairness_metrics(y_true, y_pred, gender)

    return results


def compute_fairness_metrics(y_true, y_pred, gender):
    """
    Compute all fairness metrics for a single model.

    Returns a dict of metrics with values and ethical interpretations.
    """
    # Convert to numpy for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    gender = np.array(gender)

    male_mask = gender == "Male"
    female_mask = gender == "Female"

    # ----------------------------------------------------------------
    # METRIC 1: Demographic Parity Difference
    # ----------------------------------------------------------------
    # Ethical basis: EGALITARIAN — everyone should have equal selection rates
    # regardless of group membership.
    # Value: difference in positive prediction rates between groups.
    # Ideal value: 0.0 (perfect parity)
    # Positive value: males are selected more often
    # Negative value: females are selected more often
    dp_diff = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=gender
    )

    # ----------------------------------------------------------------
    # METRIC 2: Demographic Parity Ratio (Disparate Impact Ratio)
    # ----------------------------------------------------------------
    # Ethical/Legal basis: US anti-discrimination law (80% rule / 4/5 rule)
    # The EEOC considers a ratio below 0.8 as evidence of adverse impact.
    # Value: female selection rate / male selection rate
    # Ideal value: 1.0 (identical rates)
    # Below 0.8 = legally significant adverse impact against women
    dp_ratio = demographic_parity_ratio(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=gender
    )

    # ----------------------------------------------------------------
    # METRIC 3: Equalized Odds Difference
    # ----------------------------------------------------------------
    # Ethical basis: PROCEDURAL FAIRNESS — the model's errors should be
    # equally distributed across demographic groups.
    # Measures difference in BOTH true positive rate AND false positive rate.
    # Ideal value: 0.0
    eo_diff = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=gender
    )

    # ----------------------------------------------------------------
    # METRIC 4: Equal Opportunity (True Positive Rate by Gender)
    # ----------------------------------------------------------------
    # Ethical basis: MERITOCRATIC — among actually qualified candidates,
    # both groups should have the same chance of being recommended.
    # Measures: of all candidates who SHOULD be hired, what % does the AI recommend?
    male_tpr = _true_positive_rate(y_true[male_mask], y_pred[male_mask])
    female_tpr = _true_positive_rate(y_true[female_mask], y_pred[female_mask])
    equal_opportunity_gap = male_tpr - female_tpr

    # ----------------------------------------------------------------
    # METRIC 5: False Negative Rate by Gender
    # ----------------------------------------------------------------
    # The false negative rate tells us: among qualified candidates,
    # what proportion is the AI incorrectly rejecting?
    # A higher FNR for women = the AI is failing qualified female candidates
    male_fnr = _false_negative_rate(y_true[male_mask], y_pred[male_mask])
    female_fnr = _false_negative_rate(y_true[female_mask], y_pred[female_mask])

    # ----------------------------------------------------------------
    # METRIC 6: Selection Rates by Gender
    # ----------------------------------------------------------------
    male_selection_rate = y_pred[male_mask].mean()
    female_selection_rate = y_pred[female_mask].mean()

    # ----------------------------------------------------------------
    # METRIC 7: Overall Accuracy Metrics
    # ----------------------------------------------------------------
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, zero_division=0)
    overall_recall = recall_score(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        # Core fairness metrics
        "demographic_parity_difference": round(float(dp_diff), 4),
        "disparate_impact_ratio": round(float(dp_ratio), 4),
        "equalized_odds_difference": round(float(eo_diff), 4),
        "equal_opportunity_gap": round(float(equal_opportunity_gap), 4),

        # By-gender breakdowns
        "male_selection_rate": round(float(male_selection_rate), 4),
        "female_selection_rate": round(float(female_selection_rate), 4),
        "male_true_positive_rate": round(float(male_tpr), 4),
        "female_true_positive_rate": round(float(female_tpr), 4),
        "male_false_negative_rate": round(float(male_fnr), 4),
        "female_false_negative_rate": round(float(female_fnr), 4),

        # Overall performance
        "accuracy": round(float(overall_accuracy), 4),
        "precision": round(float(overall_precision), 4),
        "recall": round(float(overall_recall), 4),
        "f1_score": round(float(overall_f1), 4),
    }


def compute_confusion_matrices(y_true, y_pred, gender):
    """
    Compute confusion matrices separately for male and female candidates.

    ETHICAL NOTE:
    Comparing confusion matrices by gender reveals whether the model's
    errors are equally distributed. If the model makes more mistakes on
    female candidates — particularly more false negatives (rejecting qualified
    women) — that is a form of measurable discrimination.
    """
    gender = np.array(gender)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm_male = confusion_matrix(
        y_true[gender == "Male"],
        y_pred[gender == "Male"]
    )
    cm_female = confusion_matrix(
        y_true[gender == "Female"],
        y_pred[gender == "Female"]
    )

    return cm_male, cm_female


def interpret_metrics(metrics: dict, model_name: str) -> list:
    """
    Return human-readable ethical interpretations of fairness metrics.
    Designed for display in the Streamlit app and Jupyter Notebook.
    """
    interpretations = []

    dp_diff = metrics["demographic_parity_difference"]
    di_ratio = metrics["disparate_impact_ratio"]
    eo_diff = metrics["equalized_odds_difference"]
    f_sel = metrics["female_selection_rate"]
    m_sel = metrics["male_selection_rate"]
    f_tpr = metrics["female_true_positive_rate"]
    m_tpr = metrics["male_true_positive_rate"]
    f_fnr = metrics["female_false_negative_rate"]

    # Demographic Parity interpretation
    if abs(dp_diff) < 0.05:
        dp_status = "✅ ACCEPTABLE"
        dp_msg = f"Selection rates are similar: {m_sel*100:.1f}% (male) vs {f_sel*100:.1f}% (female)."
    else:
        dp_status = "⚠️ CONCERN"
        direction = "favors male" if dp_diff > 0 else "favors female"
        dp_msg = f"Selection rates differ significantly: {m_sel*100:.1f}% (male) vs {f_sel*100:.1f}% (female). Gap {direction} candidates by {abs(dp_diff)*100:.1f}pp."

    interpretations.append({
        "metric": "Demographic Parity",
        "ethical_basis": "Egalitarian — equal outcomes regardless of group",
        "value": dp_diff,
        "status": dp_status,
        "interpretation": dp_msg
    })

    # Disparate Impact interpretation (EEOC 80% rule)
    if di_ratio >= 0.8:
        di_status = "✅ PASSES 80% RULE"
        di_msg = f"Disparate impact ratio: {di_ratio:.3f}. Meets the EEOC legal threshold (≥0.80)."
    else:
        di_status = "🚨 FAILS 80% RULE"
        di_msg = f"Disparate impact ratio: {di_ratio:.3f}. BELOW the EEOC 0.80 threshold — would signal adverse impact under US employment law."

    interpretations.append({
        "metric": "Disparate Impact Ratio",
        "ethical_basis": "Anti-discrimination law — EEOC 80% / 4/5 rule",
        "value": di_ratio,
        "status": di_status,
        "interpretation": di_msg
    })

    # Equal Opportunity interpretation
    if abs(eo_diff) < 0.05:
        eo_status = "✅ ACCEPTABLE"
        eo_msg = f"Qualified candidates of both genders are recommended at similar rates. TPR: {m_tpr*100:.1f}% (male) vs {f_tpr*100:.1f}% (female)."
    else:
        eo_status = "⚠️ CONCERN"
        eo_msg = f"Qualified candidates are NOT equally recognized. TPR: {m_tpr*100:.1f}% (male) vs {f_tpr*100:.1f}% (female). Qualified women are being missed at a higher rate."

    interpretations.append({
        "metric": "Equalized Odds",
        "ethical_basis": "Procedural fairness — equal error distribution across groups",
        "value": eo_diff,
        "status": eo_status,
        "interpretation": eo_msg
    })

    # False Negative Rate interpretation
    if f_fnr > 0.3:
        fnr_status = "🚨 HIGH RISK"
        fnr_msg = f"The model incorrectly rejects {f_fnr*100:.1f}% of qualified female candidates. These are real women who deserved an interview but were filtered out."
    elif f_fnr > 0.15:
        fnr_status = "⚠️ ELEVATED"
        fnr_msg = f"{f_fnr*100:.1f}% of qualified female candidates are incorrectly rejected. This represents meaningful lost opportunity."
    else:
        fnr_status = "✅ ACCEPTABLE"
        fnr_msg = f"False negative rate for female candidates is {f_fnr*100:.1f}% — within acceptable range."

    interpretations.append({
        "metric": "False Negative Rate (Women)",
        "ethical_basis": "Human cost — qualified women incorrectly rejected",
        "value": f_fnr,
        "status": fnr_status,
        "interpretation": fnr_msg
    })

    return interpretations


def _true_positive_rate(y_true, y_pred):
    """True positive rate (recall) for positive class."""
    if y_true.sum() == 0:
        return 0.0
    return ((y_true == 1) & (y_pred == 1)).sum() / (y_true == 1).sum()


def _false_negative_rate(y_true, y_pred):
    """False negative rate — proportion of actual positives incorrectly rejected."""
    if y_true.sum() == 0:
        return 0.0
    return ((y_true == 1) & (y_pred == 0)).sum() / (y_true == 1).sum()


def summarize_audit_comparison(biased_metrics: dict, fair_metrics: dict) -> pd.DataFrame:
    """
    Create a side-by-side comparison DataFrame for visualization.
    """
    key_metrics = [
        ("Demographic Parity Difference", "demographic_parity_difference", "lower is better"),
        ("Disparate Impact Ratio", "disparate_impact_ratio", "closer to 1.0 is better"),
        ("Equalized Odds Difference", "equalized_odds_difference", "lower is better"),
        ("Equal Opportunity Gap", "equal_opportunity_gap", "lower is better"),
        ("Female Selection Rate", "female_selection_rate", "should match male rate"),
        ("Male Selection Rate", "male_selection_rate", "reference"),
        ("Female False Negative Rate", "female_false_negative_rate", "lower is better"),
        ("Overall Accuracy", "accuracy", "higher is better"),
        ("Overall F1 Score", "f1_score", "higher is better"),
    ]

    rows = []
    for label, key, direction in key_metrics:
        rows.append({
            "Metric": label,
            "Biased Model": biased_metrics[key],
            "Fairness-Aware Model": fair_metrics[key],
            "Direction": direction,
            "Improvement": round(fair_metrics[key] - biased_metrics[key], 4)
        })

    return pd.DataFrame(rows)
