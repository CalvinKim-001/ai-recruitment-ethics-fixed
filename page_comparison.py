"""
page_comparison.py
------------------
Model Comparison Page — side-by-side analysis of biased vs fairness-aware model.
Includes accuracy vs. fairness tradeoff visualization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fairness_audit import summarize_audit_comparison

BIASED_COLOR = "#C62828"
FAIR_COLOR = "#2E7D32"
MALE_COLOR = "#1565C0"
FEMALE_COLOR = "#AD1457"


def render():
    st.title("🔄 Model Comparison")
    st.markdown("""
    This page places both models side by side, making the fairness improvements
    — and their tradeoffs — visible and measurable.
    """)

    if not st.session_state.get("models_trained"):
        st.warning("⚠️ Please train the models on the **Fairness Audit Dashboard** page first.")
        return

    biased_metrics = st.session_state.get("biased_metrics")
    fair_metrics = st.session_state.get("fair_metrics")

    if biased_metrics is None:
        st.warning("⚠️ Fairness metrics not yet computed. Please visit the Fairness Audit page.")
        return

    # ----------------------------------------------------------------
    # SECTION 1: Head-to-Head Scorecard
    # ----------------------------------------------------------------
    st.markdown("### ⚔️ Head-to-Head: Fairness Scorecard")

    comparison_df = summarize_audit_comparison(biased_metrics, fair_metrics)

    # Display as styled table
    def style_row(row):
        metric = row["Metric"]
        direction = row["Direction"]
        improvement = row["Improvement"]

        # Determine if improvement is actually good
        if "lower is better" in direction:
            good = improvement < -0.005
            bad = improvement > 0.005
        elif "closer to 1.0" in direction:
            good = abs(1.0 - row["Fairness-Aware Model"]) < abs(1.0 - row["Biased Model"])
            bad = not good
        elif "higher is better" in direction:
            good = improvement > 0.005
            bad = improvement < -0.005
        else:
            good = bad = False

        if good:
            return [""] * 4 + ["background-color: #e8f5e9"]
        elif bad:
            return [""] * 4 + ["background-color: #ffebee"]
        else:
            return [""] * 5

    st.dataframe(
        comparison_df.style.apply(style_row, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------------------------------
    # SECTION 2: Radar / Spider Chart of Fairness Metrics
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🕸️ Fairness Profile Comparison")

    metrics_for_radar = {
        "Demographic\nParity": (
            1 - abs(biased_metrics["demographic_parity_difference"]),
            1 - abs(fair_metrics["demographic_parity_difference"])
        ),
        "Disparate\nImpact": (
            min(biased_metrics["disparate_impact_ratio"], 1.0),
            min(fair_metrics["disparate_impact_ratio"], 1.0)
        ),
        "Equal\nOpportunity": (
            1 - abs(biased_metrics["equal_opportunity_gap"]),
            1 - abs(fair_metrics["equal_opportunity_gap"])
        ),
        "Equalized\nOdds": (
            1 - abs(biased_metrics["equalized_odds_difference"]),
            1 - abs(fair_metrics["equalized_odds_difference"])
        ),
        "Low FNR\n(Women)": (
            1 - biased_metrics["female_false_negative_rate"],
            1 - fair_metrics["female_false_negative_rate"]
        ),
        "Selection\nParity": (
            1 - abs(biased_metrics["male_selection_rate"] - biased_metrics["female_selection_rate"]),
            1 - abs(fair_metrics["male_selection_rate"] - fair_metrics["female_selection_rate"])
        ),
    }

    labels = list(metrics_for_radar.keys())
    biased_vals = [v[0] for v in metrics_for_radar.values()]
    fair_vals = [v[1] for v in metrics_for_radar.values()]

    # Normalize to 0–1
    biased_vals = np.clip(biased_vals, 0, 1)
    fair_vals = np.clip(fair_vals, 0, 1)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    biased_vals_plot = biased_vals.tolist() + biased_vals[:1].tolist()
    fair_vals_plot = fair_vals.tolist() + fair_vals[:1].tolist()

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, biased_vals_plot, "o-", linewidth=2, color=BIASED_COLOR, label="Biased Model")
    ax.fill(angles, biased_vals_plot, alpha=0.15, color=BIASED_COLOR)
    ax.plot(angles, fair_vals_plot, "s-", linewidth=2, color=FAIR_COLOR, label="Fairness-Aware Model")
    ax.fill(angles, fair_vals_plot, alpha=0.15, color=FAIR_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Fairness Profile\n(Higher = More Fair)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.grid(color="grey", linestyle="--", alpha=0.3)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("""
        **Reading the chart:**

        Each axis represents one fairness metric, normalized so that
        **higher (closer to the outer edge) = more fair**.

        The **red shape** (biased model) shows the fairness profile of
        a model trained without any corrections.

        The **green shape** (fairness-aware model) shows how fairness
        mitigation improves performance across multiple metrics.

        Notice: the green shape is almost always larger — meaning the
        fairness-aware model performs better on fairness dimensions
        while maintaining reasonable predictive performance.

        **No model reaches a perfect outer circle.** This is honest:
        perfect fairness across all metrics simultaneously is mathematically
        impossible. Responsible AI governance acknowledges this.
        """)

    # ----------------------------------------------------------------
    # SECTION 3: Accuracy vs Fairness Tradeoff
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚖️ The Core Tradeoff: Accuracy vs. Fairness")

    st.markdown("""
    This is one of the most important findings in modern AI ethics research:
    **improving fairness can reduce accuracy, and improving accuracy can reduce fairness.**

    The chart below visualizes where each model sits on the accuracy-fairness spectrum.
    There is no model in the top-right corner (perfect accuracy AND perfect fairness) —
    and any system claiming to be there should be treated with skepticism.
    """)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot models
    biased_acc = biased_metrics["accuracy"]
    fair_acc = fair_metrics["accuracy"]

    # Fairness score = 1 - |demographic parity diff| (simple composite)
    biased_fairness = 1 - abs(biased_metrics["demographic_parity_difference"])
    fair_fairness = 1 - abs(fair_metrics["demographic_parity_difference"])

    ax.scatter(biased_acc, biased_fairness, s=300, color=BIASED_COLOR, zorder=5,
              marker="o", label="Biased Baseline Model", edgecolors="darkred", linewidth=1.5)
    ax.scatter(fair_acc, fair_fairness, s=300, color=FAIR_COLOR, zorder=5,
              marker="s", label="Fairness-Aware Model", edgecolors="darkgreen", linewidth=1.5)

    # Annotate
    ax.annotate(
        f"Biased Model\nAcc: {biased_acc:.3f}\nFairness: {biased_fairness:.3f}",
        xy=(biased_acc, biased_fairness),
        xytext=(biased_acc - 0.06, biased_fairness - 0.06),
        arrowprops=dict(arrowstyle="->", color=BIASED_COLOR),
        fontsize=9, color=BIASED_COLOR, fontweight="500"
    )
    ax.annotate(
        f"Fairness-Aware Model\nAcc: {fair_acc:.3f}\nFairness: {fair_fairness:.3f}",
        xy=(fair_acc, fair_fairness),
        xytext=(fair_acc + 0.01, fair_fairness + 0.03),
        arrowprops=dict(arrowstyle="->", color=FAIR_COLOR),
        fontsize=9, color=FAIR_COLOR, fontweight="500"
    )

    # Ideal point
    ax.scatter(1.0, 1.0, s=200, color="gold", zorder=3, marker="*",
              label="Ideal (unachievable)", alpha=0.7)
    ax.annotate("Ideal\n(unachievable)", xy=(1.0, 1.0), xytext=(0.96, 0.95),
               fontsize=8, color="goldenrod", style="italic")

    # Quadrant labels
    ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0.75, color="gray", linestyle="--", alpha=0.3)
    ax.text(0.65, 0.95, "High Fairness\nLow Accuracy", fontsize=8, color="gray",
           ha="center", style="italic")
    ax.text(0.88, 0.60, "Low Fairness\nHigh Accuracy", fontsize=8, color="gray",
           ha="center", style="italic")

    ax.set_xlabel("Model Accuracy (Predictive Performance)", fontsize=11)
    ax.set_ylabel("Fairness Score\n(1 − |Demographic Parity Difference|)", fontsize=11)
    ax.set_title("Accuracy vs. Fairness Tradeoff Space\nNo system achieves perfect scores on both dimensions",
                fontweight="bold")
    ax.legend(loc="lower left")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig)
    plt.close()

    # ----------------------------------------------------------------
    # SECTION 4: Ethical Discussion of Tradeoff
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💭 What Does This Tradeoff Mean Ethically?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **The utilitarian argument for accuracy:**
        A more accurate model makes better predictions overall — which could benefit
        all candidates by making the hiring process more efficient. If accuracy drops
        significantly, the model becomes less useful, and organizations might abandon
        it entirely rather than use a fairer but weaker system.
        """)

        st.markdown("""
        **The deontological argument for fairness:**
        Regardless of aggregate outcomes, every candidate has a right to be evaluated
        without discrimination. A model that is more "accurate" but discriminates against
        women is not making better decisions — it is making faster discriminatory decisions.
        The efficiency gain does not justify the rights violation.
        """)

    with col2:
        st.markdown("""
        **The legal argument:**
        Under the EEOC disparate impact doctrine (and increasingly under EU AI Act
        regulations), accuracy is not a defense for discrimination. An organization
        cannot claim that a biased hiring system is acceptable because it is accurate
        overall — if it disproportionately disadvantages a protected group, it is
        legally and ethically problematic regardless of its accuracy score.
        """)

        st.markdown("""
        **Our position:**
        This project does not claim to have resolved the tradeoff — because it cannot
        be fully resolved. What we demonstrate is that **measurable improvements in
        fairness are achievable**, that those improvements have modest rather than
        catastrophic accuracy costs, and that responsible organizations should be
        willing to accept those costs as a matter of ethics and governance.
        """)

    st.markdown("""
    <div class='ethical-note'>
        <strong>🔑 Key takeaway:</strong> The choice of which model to deploy is itself
        a moral decision. Choosing the biased model for its higher accuracy is not
        a "neutral" technical choice — it is an active decision to prioritize prediction
        performance over equitable treatment. Organizations that make that choice should
        make it explicitly, transparently, and with accountability to affected candidates.
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # SECTION 5: Regulatory Context
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🏛️ Regulatory Context: What the Law Says")

    reg_data = {
        "Regulation": [
            "EU AI Act (2024)",
            "NYC Local Law 144 (2023)",
            "EEOC Guidelines (US)",
            "UK Equality Act (2010)",
        ],
        "Classification": [
            "HIGH RISK — mandatory requirements",
            "Bias audit required before use",
            "Disparate impact doctrine",
            "Indirect discrimination",
        ],
        "Key Requirement": [
            "Human oversight, bias auditing, transparency documentation, registration",
            "Annual bias audit by independent auditor, public disclosure of results",
            "80% rule: selection rates must not fall below 80% of highest group",
            "Employers must justify practices with disproportionate impact",
        ],
        "Implication for This Project": [
            "Our audit log and fairness dashboard demonstrate the transparency this law requires",
            "Our disparate impact ratio metric operationalizes exactly this law's requirement",
            "Our disparate impact ratio must meet ≥0.80 threshold — see Fairness Audit page",
            "Fairness-aware model is designed to reduce practices with disproportionate impact",
        ],
    }

    st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)
