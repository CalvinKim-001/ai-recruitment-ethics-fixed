"""
page_fairness.py
----------------
Fairness Audit Dashboard — trains both models and computes all fairness metrics.
This is the analytical core of the application.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_generator import generate_hiring_dataset, get_dataset_summary
from models import (
    prepare_data, train_biased_model, train_fairness_aware_model,
    FEATURE_COLUMNS
)
from fairness_audit import (
    compute_fairness_metrics, compute_confusion_matrices,
    interpret_metrics, summarize_audit_comparison
)

# Color palette
BIASED_COLOR = "#C62828"
FAIR_COLOR = "#2E7D32"
MALE_COLOR = "#1565C0"
FEMALE_COLOR = "#AD1457"


def render():
    st.title("📊 Fairness Audit Dashboard")
    st.markdown("""
    This page trains both recruitment models and performs a complete fairness audit.
    The metrics computed here form the empirical foundation of the project's ethical argument.
    """)

    # ----------------------------------------------------------------
    # STEP 1: Initialize / Train Models
    # ----------------------------------------------------------------
    st.markdown("### Step 1 — Initialize Models")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        n_candidates = st.slider(
            "Dataset size (number of synthetic candidates):",
            min_value=500, max_value=2000, value=1000, step=100
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        train_button = st.button("🚀 Train Both Models", type="primary", use_container_width=True)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.models_trained:
            st.success("✅ Ready")

    if train_button:
        with st.spinner("Generating synthetic dataset and training models... (~10 seconds)"):
            # Generate dataset
            df = generate_hiring_dataset(n_candidates)
            st.session_state.dataset = df

            # Prepare train/test split
            X_train, X_test, y_train, y_test, g_train, g_test = prepare_data(df)
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.g_test = g_test

            # Train biased model
            biased_model = train_biased_model(X_train, y_train)
            st.session_state.biased_model = biased_model

            # Train fairness-aware model
            fair_model, scaler = train_fairness_aware_model(X_train, y_train, g_train)
            st.session_state.fair_model = fair_model
            st.session_state.fair_scaler = scaler

            # Get predictions
            y_pred_biased = biased_model.predict(X_test)

            X_test_scaled = scaler.transform(X_test)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)
            y_pred_fair = fair_model.predict(X_test_scaled_df)

            st.session_state.y_pred_biased = y_pred_biased
            st.session_state.y_pred_fair = y_pred_fair
            st.session_state.models_trained = True

        st.success("✅ Models trained successfully!")

    if not st.session_state.models_trained:
        st.info("👆 Click **Train Both Models** to begin the fairness audit.")
        return

    # ----------------------------------------------------------------
    # STEP 2: Dataset Overview
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 2 — Dataset: The Imbalance We Built In")

    st.markdown("""
    <div class='ethical-note'>
        <strong>Why is the data intentionally imbalanced?</strong><br>
        We deliberately built gender imbalance into this dataset — 72% male candidates
        with higher historical hire rates. This is not our belief about who deserves to be
        hired. It simulates the historical reality of tech hiring that Amazon's AI learned from.
        The model's subsequent bias is caused entirely by this imbalance. This is the point.
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.dataset
    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", f"{len(df):,}")
    with col2:
        st.metric("Male Candidates", f"{summary['Male']['count']:,}",
                  delta=f"{summary['Male']['pct_of_dataset']}% of dataset")
    with col3:
        st.metric("Female Candidates", f"{summary['Female']['count']:,}",
                  delta=f"{summary['Female']['pct_of_dataset']}% of dataset",
                  delta_color="inverse")
    with col4:
        hire_gap = summary['Male']['hire_rate'] - summary['Female']['hire_rate']
        st.metric("Historical Hire Rate Gap",
                  f"{hire_gap:.1f}pp",
                  help="Percentage point gap in hire rates between male and female candidates in the training data")

    # Historical hire rate comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gender distribution
    genders = ["Male", "Female"]
    counts = [summary[g]["count"] for g in genders]
    bars = axes[0].bar(genders, counts, color=[MALE_COLOR, FEMALE_COLOR], alpha=0.85, width=0.5)
    axes[0].set_title("Gender Distribution in Training Data", fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Candidates")
    for bar, count, pct in zip(bars, counts, [summary[g]["pct_of_dataset"] for g in genders]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{count}\n({pct}%)", ha="center", fontsize=10, fontweight="500")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_ylim(0, max(counts) * 1.15)

    # Historical hire rates
    hire_rates = [summary[g]["hire_rate"] for g in genders]
    bars2 = axes[1].bar(genders, hire_rates, color=[MALE_COLOR, FEMALE_COLOR], alpha=0.85, width=0.5)
    axes[1].set_title("Historical Hire Rate by Gender\n(in Training Data — intentionally biased)",
                      fontweight="bold", pad=12)
    axes[1].set_ylabel("Hire Rate (%)")
    axes[1].set_ylim(0, max(hire_rates) * 1.2)
    for bar, rate in zip(bars2, hire_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{rate}%", ha="center", fontsize=11, fontweight="600")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Add annotation arrow showing gap
    axes[1].annotate("", xy=(1, hire_rates[1]), xytext=(1, hire_rates[0]),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    axes[1].text(1.15, (hire_rates[0] + hire_rates[1])/2,
                f"Gap:\n{hire_gap:.1f}pp", fontsize=9, va="center", color="black")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ----------------------------------------------------------------
    # STEP 3: Fairness Metrics Comparison
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 3 — Fairness Audit: Biased vs. Fairness-Aware Model")

    y_test = np.array(st.session_state.y_test)
    y_pred_biased = np.array(st.session_state.y_pred_biased)
    y_pred_fair = np.array(st.session_state.y_pred_fair)
    g_test = np.array(st.session_state.g_test)

    biased_metrics = compute_fairness_metrics(y_test, y_pred_biased, g_test)
    fair_metrics = compute_fairness_metrics(y_test, y_pred_fair, g_test)

    # Store for other pages
    st.session_state.biased_metrics = biased_metrics
    st.session_state.fair_metrics = fair_metrics

    # Key metrics side by side
    st.markdown("#### Key Fairness Metrics")

    metric_display = [
        ("Demographic Parity\nDifference", "demographic_parity_difference",
         "Closer to 0 is fairer", False),
        ("Disparate Impact\nRatio", "disparate_impact_ratio",
         "Closer to 1.0 is fairer. ≥0.80 = passes EEOC rule", True),
        ("Equalized Odds\nDifference", "equalized_odds_difference",
         "Closer to 0 is fairer", False),
        ("Female Selection\nRate", "female_selection_rate",
         "Should be similar to male rate", True),
        ("Female False\nNegative Rate", "female_false_negative_rate",
         "Lower = fewer qualified women incorrectly rejected", False),
        ("Overall\nAccuracy", "accuracy",
         "Prediction accuracy (note: fairness may reduce accuracy slightly)", True),
    ]

    cols = st.columns(3)
    for i, (label, key, help_text, higher_is_better) in enumerate(metric_display):
        with cols[i % 3]:
            b_val = biased_metrics[key]
            f_val = fair_metrics[key]
            delta = f_val - b_val
            delta_str = f"{delta:+.4f}"

            # For metrics where lower is better, invert delta color
            if not higher_is_better:
                delta_color = "normal" if delta < 0 else "inverse"
            else:
                delta_color = "normal" if delta > 0 else "inverse"

            st.markdown(f"<div style='font-size:0.8rem; font-weight:600; color:#555; margin-bottom:4px;'>{label}</div>", unsafe_allow_html=True)
            col_b, col_f = st.columns(2)
            with col_b:
                st.markdown(f"""
                <div style='background:#ffebee; border-radius:8px; padding:10px; text-align:center; border:1px solid #ef9a9a;'>
                    <div style='font-size:0.7rem; color:#c62828; font-weight:600;'>BIASED</div>
                    <div style='font-size:1.3rem; font-weight:700; color:#c62828;'>{b_val:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_f:
                st.markdown(f"""
                <div style='background:#e8f5e9; border-radius:8px; padding:10px; text-align:center; border:1px solid #a5d6a7;'>
                    <div style='font-size:0.7rem; color:#2e7d32; font-weight:600;'>FAIR</div>
                    <div style='font-size:1.3rem; font-weight:700; color:#2e7d32;'>{f_val:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.caption(help_text)
            st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # STEP 4: Selection Rates Visualization
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 4 — Selection Rates by Gender & Model")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    male_rates = [biased_metrics["male_selection_rate"]*100, fair_metrics["male_selection_rate"]*100]
    female_rates = [biased_metrics["female_selection_rate"]*100, fair_metrics["female_selection_rate"]*100]
    x = np.arange(2)
    width = 0.3

    axes[0].bar(x - width/2, male_rates, width, label="Male Candidates",
               color=MALE_COLOR, alpha=0.85)
    axes[0].bar(x + width/2, female_rates, width, label="Female Candidates",
               color=FEMALE_COLOR, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Biased\nBaseline", "Fairness-Aware\nModel"])
    axes[0].set_ylabel("Selection Rate (%)")
    axes[0].set_title("AI Recommendation Rate by Gender\nBefore and After Fairness Mitigation",
                      fontweight="bold")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for i, (m, f) in enumerate(zip(male_rates, female_rates)):
        axes[0].text(i - width/2, m + 0.5, f"{m:.1f}%", ha="center", fontsize=9)
        axes[0].text(i + width/2, f + 0.5, f"{f:.1f}%", ha="center", fontsize=9)

    # TPR (Equal Opportunity) comparison
    male_tpr = [biased_metrics["male_true_positive_rate"]*100, fair_metrics["male_true_positive_rate"]*100]
    female_tpr = [biased_metrics["female_true_positive_rate"]*100, fair_metrics["female_true_positive_rate"]*100]

    axes[1].bar(x - width/2, male_tpr, width, label="Male Candidates",
               color=MALE_COLOR, alpha=0.85)
    axes[1].bar(x + width/2, female_tpr, width, label="Female Candidates",
               color=FEMALE_COLOR, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Biased\nBaseline", "Fairness-Aware\nModel"])
    axes[1].set_ylabel("True Positive Rate (%)")
    axes[1].set_title("Equal Opportunity: Qualified Candidates Correctly Recommended\n(True Positive Rate by Gender)",
                      fontweight="bold")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for i, (m, f) in enumerate(zip(male_tpr, female_tpr)):
        axes[1].text(i - width/2, m + 0.5, f"{m:.1f}%", ha="center", fontsize=9)
        axes[1].text(i + width/2, f + 0.5, f"{f:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ----------------------------------------------------------------
    # STEP 5: Confusion Matrices
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 5 — Confusion Matrices by Gender")

    st.markdown("""
    Confusion matrices show what *kinds* of errors the model makes — and whether
    those errors are distributed equally between men and women.
    A **false negative** means a qualified candidate was incorrectly rejected.
    If the model has more false negatives for women than men, that is measurable discrimination.
    """)

    tab_biased, tab_fair = st.tabs(["🔴 Biased Baseline Model", "🟢 Fairness-Aware Model"])

    for tab, y_pred, model_name in [
        (tab_biased, y_pred_biased, "Biased Model"),
        (tab_fair, y_pred_fair, "Fairness-Aware Model"),
    ]:
        with tab:
            cm_male, cm_female = compute_confusion_matrices(y_test, y_pred, g_test)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, cm, gender_label, color in [
                (axes[0], cm_male, "Male Candidates", MALE_COLOR),
                (axes[1], cm_female, "Female Candidates", FEMALE_COLOR),
            ]:
                im = ax.imshow(cm, interpolation="nearest",
                              cmap=plt.cm.Blues if gender_label == "Male Candidates" else plt.cm.RdPu)
                ax.set_title(f"{gender_label}", fontweight="bold", color=color)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Rejected", "Hired"])
                ax.set_yticklabels(["Rejected", "Hired"])

                thresh = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        label = "TN" if (i==0 and j==0) else "FP" if (i==0 and j==1) else "FN" if (i==1 and j==0) else "TP"
                        ax.text(j, i, f"{cm[i, j]}\n({label})",
                               ha="center", va="center", fontsize=12,
                               color="white" if cm[i, j] > thresh else "black",
                               fontweight="600")

            plt.suptitle(f"{model_name} — Confusion Matrices by Gender", fontweight="bold", y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # FN analysis
            if len(cm_female) > 1:
                fn_female = cm_female[1, 0]
                fn_male = cm_male[1, 0]
                total_female_pos = cm_female[1].sum()
                total_male_pos = cm_male[1].sum()
                fnr_female = fn_female / total_female_pos if total_female_pos > 0 else 0
                fnr_male = fn_male / total_male_pos if total_male_pos > 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("FN Rate — Female Candidates", f"{fnr_female*100:.1f}%",
                             help="Proportion of qualified female candidates incorrectly rejected")
                with col2:
                    st.metric("FN Rate — Male Candidates", f"{fnr_male*100:.1f}%")
                with col3:
                    gap = fnr_female - fnr_male
                    st.metric("FN Rate Gap (F−M)", f"{gap*100:.1f}pp",
                             delta=f"{'Bias against women' if gap > 0.02 else 'Acceptable range'}",
                             delta_color="inverse" if gap > 0.02 else "normal")

    # ----------------------------------------------------------------
    # STEP 6: Metric Interpretations
    # ----------------------------------------------------------------
    st.divider()
    st.markdown("### Step 6 — Ethical Interpretations")

    tab1, tab2 = st.tabs(["🔴 Biased Model Interpretations", "🟢 Fairness-Aware Model Interpretations"])

    with tab1:
        interpretations = interpret_metrics(biased_metrics, "Biased Model")
        for interp in interpretations:
            border_color = "#c62828" if any(x in interp["status"] for x in ["CONCERN", "FAILS", "HIGH"]) else "#2e7d32"
            st.markdown(f"""
            <div style='background-color:#F8F9FA; border-radius:8px; padding:16px; margin:8px 0; border-left:6px solid {border_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <div style='color:#000000; font-weight:900; font-size:1.1rem; margin-bottom:6px;'>{interp["metric"]}</div>
                <div style='color:#444444; font-size:0.85rem; margin-bottom:12px;'>Ethical basis: {interp["ethical_basis"]}</div>
                <div style='color:#000000; font-weight:800; margin-bottom:6px;'>{interp["status"]}</div>
                <div style='color:#111111; font-size:0.95rem; line-height:1.5;'>{interp["interpretation"]}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        interpretations_fair = interpret_metrics(fair_metrics, "Fairness-Aware Model")
        for interp in interpretations_fair:
            border_color = "#2e7d32" if any(x in interp["status"] for x in ["ACCEPTABLE", "PASSES"]) else "#f9a825"
            st.markdown(f"""
            <div style='background-color:#F8F9FA; border-radius:8px; padding:16px; margin:8px 0; border-left:6px solid {border_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <div style='color:#000000; font-weight:900; font-size:1.1rem; margin-bottom:6px;'>{interp["metric"]}</div>
                <div style='color:#444444; font-size:0.85rem; margin-bottom:12px;'>Ethical basis: {interp["ethical_basis"]}</div>
                <div style='color:#000000; font-weight:800; margin-bottom:6px;'>{interp["status"]}</div>
                <div style='color:#111111; font-size:0.95rem; line-height:1.5;'>{interp["interpretation"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Fairness tradeoff note
    st.markdown("""
    <div class='ethical-note'>
        <strong>⚖️ The Fairness Tradeoff:</strong> Notice that the fairness-aware model
        may show slightly lower overall accuracy. This is not a bug — it is an intentional
        tradeoff. Improving fairness sometimes requires accepting a modest reduction in
        raw predictive performance. This mirrors the real-world challenge faced by every
        organization that tries to build fairer AI systems. There is no free lunch in
        AI ethics.
    </div>
    """, unsafe_allow_html=True)