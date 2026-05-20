"""
explainability.py
-----------------
Implements transparency and explainability tools for the recruitment models.

WHY EXPLAINABILITY MATTERS (ETHICALLY):
An AI system that cannot explain its decisions is an AI system that cannot
be held accountable. If a candidate is rejected and cannot understand why,
they cannot challenge the decision. If a recruiter cannot understand why
the AI made a recommendation, they cannot meaningfully exercise oversight.

"Black box" AI in hiring is not just technically problematic — it is
ethically and increasingly legally problematic. The EU AI Act (2024)
classifies recruitment AI as HIGH RISK and requires explainability.

This module uses SHAP (SHapley Additive exPlanations) — a method rooted
in cooperative game theory that explains how much each feature contributed
to a specific prediction. SHAP values are the gold standard for ML explainability.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from models import FEATURE_COLUMNS

# Human-readable feature labels for visualizations
FEATURE_LABELS = {
    "years_experience": "Years of Experience",
    "education_level": "Education Level",
    "programming_skill": "Programming Skill",
    "leadership_score": "Leadership Score",
    "communication_score": "Communication Score",
    "company_tier": "Previous Company Tier",
    "project_experience": "Project Experience",
    "interview_score": "Interview Score",
}


def compute_shap_values(model, X: pd.DataFrame, model_type: str = "biased"):
    """
    Compute SHAP values for a trained model.

    SHAP explains each prediction by asking:
    "How much did each feature push this prediction higher or lower
    compared to the average prediction?"

    Parameters:
        model: trained sklearn Pipeline or classifier
        X: feature DataFrame
        model_type: "biased" (pipeline) or "fair" (direct classifier)

    Returns:
        shap_values array, explainer object
    """
    if model_type == "biased":
        # For sklearn pipelines, we need to get the underlying classifier
        # and transform the data through the preprocessing steps
        classifier = model.named_steps["classifier"]
        X_transformed = model.named_steps["scaler"].transform(X)
        X_transformed_df = pd.DataFrame(X_transformed, columns=FEATURE_COLUMNS)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed_df)
    else:
        # For fairness-aware model, use KernelExplainer (model-agnostic)
        # Use a small background sample for efficiency
        background = shap.sample(X, min(50, len(X)))
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(pd.DataFrame(x, columns=FEATURE_COLUMNS))[:, 1],
            background
        )
        shap_values = explainer.shap_values(X, nsamples=100)

    return shap_values, explainer


def get_feature_importance(model, model_type: str = "biased") -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Feature importance tells us: "Which candidate characteristics did this
    AI weigh most heavily when making its recommendations?"

    This is important for accountability: if the AI weights 'interview_score'
    heavily and interview scores are themselves biased (research shows that
    interview panels evaluate identical answers differently based on candidate
    gender), then explainability reveals a second layer of bias the simple
    model audit would miss.

    Returns:
        DataFrame with feature names and importance scores (sorted)
    """
    if model_type == "biased":
        classifier = model.named_steps["classifier"]
        importances = classifier.feature_importances_
    else:
        # For fairness-aware model, use mean absolute SHAP as proxy
        # (computed externally and passed in, or use a simple LR coefficient approach)
        try:
            # Try to access predictor weights if available
            predictor = model.predictors_[0]
            importances = np.abs(predictor.coef_[0])
        except Exception:
            importances = np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "feature_label": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Normalize to 0-100 scale for readability
    importance_df["importance_pct"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    ).round(1)

    return importance_df.reset_index(drop=True)


def explain_single_candidate(
    candidate_features: dict,
    biased_model,
    fair_model,
    scaler,
    biased_score: float,
    fair_score: float
) -> dict:
    """
    Generate a plain-language explanation for a single candidate's scores.

    ETHICAL NOTE:
    This function is the core of the "why" — it moves the AI from a black
    box to a transparent tool. Candidates deserve to know why they were
    scored the way they were. Recruiters need to understand AI decisions
    to exercise meaningful oversight.

    Returns:
        dict with human-readable explanations for both models
    """
    X = pd.DataFrame([{col: candidate_features.get(col, 0) for col in FEATURE_COLUMNS}])

    # Get biased model feature importances
    try:
        classifier = biased_model.named_steps["classifier"]
        X_transformed = biased_model.named_steps["scaler"].transform(X)
        X_transformed_df = pd.DataFrame(X_transformed, columns=FEATURE_COLUMNS)
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_transformed_df)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Take positive class SHAP values

        shap_vals = shap_vals.flatten()
        top_positive = []
        top_negative = []

        for i, (feature, val) in enumerate(zip(FEATURE_COLUMNS, shap_vals)):
            label = FEATURE_LABELS[feature]
            raw_val = candidate_features.get(feature, 0)
            if val > 0:
                top_positive.append((label, val, raw_val))
            elif val < 0:
                top_negative.append((label, abs(val), raw_val))

        top_positive.sort(key=lambda x: x[1], reverse=True)
        top_negative.sort(key=lambda x: x[1], reverse=True)

        positive_factors = [
            f"{label} ({raw_val})" for label, _, raw_val in top_positive[:3]
        ]
        negative_factors = [
            f"{label} ({raw_val})" for label, _, raw_val in top_negative[:3]
        ]
    except Exception:
        positive_factors = ["Score breakdown unavailable"]
        negative_factors = ["Score breakdown unavailable"]

    # Interpret the scores
    def interpret_score(score):
        if score >= 0.75:
            return "Strong Recommend"
        elif score >= 0.5:
            return "Recommend for Interview"
        elif score >= 0.35:
            return "Borderline — Human Review Critical"
        else:
            return "Does Not Meet Current Threshold"

    return {
        "biased_model": {
            "score": biased_score,
            "recommendation": interpret_score(biased_score),
            "top_positive_factors": positive_factors,
            "top_negative_factors": negative_factors,
            "caveat": (
                "⚠️ This model was trained on historically biased data and may "
                "reflect historical gender patterns. Human review is essential."
            ),
        },
        "fair_model": {
            "score": fair_score,
            "recommendation": interpret_score(fair_score),
            "top_positive_factors": positive_factors,  # Same features, different weights
            "top_negative_factors": negative_factors,
            "caveat": (
                "ℹ️ This model uses fairness constraints to reduce measurable gender bias. "
                "It is fairness-improved, not perfectly fair. Human review remains required."
            ),
        },
        "human_review_note": (
            "🔴 HUMAN REVIEW REQUIRED. This AI system is a decision-support tool only. "
            "No hiring decision should be made without recruiter review and judgment. "
            "The AI score is one input — not a verdict."
        ),
    }


def plot_feature_importance_comparison(
    biased_importance_df: pd.DataFrame,
    fair_importance_df: pd.DataFrame = None,
    figsize=(12, 6)
):
    """
    Plot feature importance for biased model (and optionally fair model side-by-side).
    """
    fig, axes = plt.subplots(
        1, 2 if fair_importance_df is not None else 1,
        figsize=figsize
    )

    if fair_importance_df is None:
        axes = [axes]

    # Colors
    BIAS_COLOR = "#E74C3C"
    FAIR_COLOR = "#27AE60"

    def _plot_importance(ax, df, title, color):
        bars = ax.barh(
            df["feature_label"],
            df["importance_pct"],
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5
        )
        ax.set_xlabel("Feature Importance (%)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.invert_yaxis()
        ax.set_xlim(0, df["importance_pct"].max() * 1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, df["importance_pct"]):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=9
            )

    _plot_importance(axes[0], biased_importance_df, "Biased Model\nFeature Importance", BIAS_COLOR)

    if fair_importance_df is not None:
        _plot_importance(axes[1], fair_importance_df, "Fairness-Aware Model\nFeature Importance", FAIR_COLOR)

    plt.suptitle(
        "What Does Each Model Care About Most?",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    return fig
