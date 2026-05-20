"""
page_evaluation.py
------------------
Candidate Evaluation Dashboard — interactive resume input + matched resume pairs experiment.
This is the most visceral demonstration: identical resumes, different scores.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import FEATURE_COLUMNS
from resume_pairs import RESUME_PAIRS, get_all_pairs_dataframe
from audit_log import log_evaluation, log_recruiter_override
from explainability import explain_single_candidate

MALE_COLOR = "#1565C0"
FEMALE_COLOR = "#AD1457"
BIASED_COLOR = "#C62828"
FAIR_COLOR = "#2E7D32"


def _check_models():
    if not st.session_state.get("models_trained"):
        st.warning("⚠️ Models are not yet trained. Please visit **📊 Fairness Audit Dashboard** first to initialize the models.")
        return False
    return True


def render():
    st.title("🔬 Candidate Evaluation")

    tab1, tab2 = st.tabs([
        "🧪 Gender Signals Experiment — Matched Resume Pairs",
        "👤 Evaluate a Custom Candidate",
    ])

    # ================================================================
    # TAB 1: MATCHED RESUME PAIRS EXPERIMENT
    # ================================================================
    with tab1:
        st.markdown("""
        ## The Core Experiment: Identical Qualifications, Different Scores

        This is the most direct test of gender bias in AI recruitment systems.
        We present **10 pairs of candidates** — each pair has **completely identical**
        qualifications. The only differences are:
        - **Name** (gendered)
        - **University** (some pairs use all-women's colleges)
        - **Activity description** (e.g., "Women's Chess Club" vs "Chess Club")

        If the biased model gives different scores to candidates with identical
        qualifications, that score difference IS gender discrimination — regardless
        of whether the word "gender" appears anywhere in the model's inputs.
        """)

        st.markdown("""
        <div class='bias-alert'>
            🚨 <strong>This replicates exactly what Amazon's system did.</strong>
            The AI penalized the word "women's" in activity descriptions and
            penalized graduates of all-women's colleges — not because these signals
            indicated lesser qualifications, but because they were underrepresented
            in historical hiring data.
        </div>
        """, unsafe_allow_html=True)

        if not _check_models():
            return

        biased_model = st.session_state.biased_model
        fair_model = st.session_state.fair_model
        scaler = st.session_state.fair_scaler

        # Score all pairs
        pairs_results = []
        for pair in RESUME_PAIRS:
            for role in ["male", "female"]:
                candidate = pair[role]
                X = pd.DataFrame([{col: candidate.get(col, 0) for col in FEATURE_COLUMNS}])
                biased_score = biased_model.predict_proba(X)[0][1]

                X_scaled = scaler.transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
                fair_score = fair_model.predict_proba(X_scaled_df)[0][1]

                pairs_results.append({
                    "pair_id": pair["pair_id"],
                    "scenario": pair["scenario"],
                    "narrative": pair["narrative"],
                    "name": candidate["name"],
                    "university": candidate["university"],
                    "activity": candidate["activity"],
                    "gender": "Male" if role == "male" else "Female",
                    "biased_score": round(float(biased_score), 4),
                    "fair_score": round(float(fair_score), 4),
                })

        results_df = pd.DataFrame(pairs_results)

        # Compute differentials
        differentials = []
        for pair_id in range(1, 11):
            pair_data = results_df[results_df["pair_id"] == pair_id]
            male_row = pair_data[pair_data["gender"] == "Male"].iloc[0]
            female_row = pair_data[pair_data["gender"] == "Female"].iloc[0]
            biased_gap = male_row["biased_score"] - female_row["biased_score"]
            fair_gap = male_row["fair_score"] - female_row["fair_score"]
            differentials.append({
                "pair_id": pair_id,
                "scenario": male_row["scenario"],
                "narrative": male_row["narrative"],
                "male_name": male_row["name"],
                "female_name": female_row["name"],
                "male_university": male_row["university"],
                "female_university": female_row["university"],
                "male_activity": male_row["activity"],
                "female_activity": female_row["activity"],
                "male_biased": male_row["biased_score"],
                "female_biased": female_row["biased_score"],
                "biased_gap": round(biased_gap, 4),
                "male_fair": male_row["fair_score"],
                "female_fair": female_row["fair_score"],
                "fair_gap": round(fair_gap, 4),
                "gap_reduction": round(abs(biased_gap) - abs(fair_gap), 4),
            })
        diff_df = pd.DataFrame(differentials)

        # ── SUMMARY CHART ──────────────────────────────────────────
        st.markdown("### 📊 Score Gap Summary: All 10 Pairs")
        st.markdown("""
        The chart below shows the **score gap** between identical male and female candidates
        for both models. A positive gap means the male candidate scored higher.
        **Larger bars = more discrimination. Smaller bars = less discrimination.**
        """)

        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(10)
        width = 0.35

        bars_b = ax.bar(x - width/2, diff_df["biased_gap"] * 100, width,
                       label="Biased Model Gap", color=BIASED_COLOR, alpha=0.85)
        bars_f = ax.bar(x + width/2, diff_df["fair_gap"] * 100, width,
                       label="Fairness-Aware Model Gap", color=FAIR_COLOR, alpha=0.85)

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Resume Pair (1–10)")
        ax.set_ylabel("Score Gap: Male − Female (%)")
        ax.set_title("Gender Score Gaps — Identical Qualifications, Only Gendered Signals Differ\n"
                     "Positive = Male scored higher | Closer to 0 = More equitable",
                     fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Pair {i+1}" for i in range(10)], rotation=30, ha="right")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate max gap
        max_gap_idx = diff_df["biased_gap"].abs().idxmax()
        max_gap_val = diff_df.loc[max_gap_idx, "biased_gap"] * 100
        ax.annotate(f"Largest gap:\n{max_gap_val:.1f}pp",
                   xy=(max_gap_idx - width/2, max_gap_val),
                   xytext=(max_gap_idx - width/2 + 1.2, max_gap_val + 1),
                   arrowprops=dict(arrowstyle="->", color=BIASED_COLOR),
                   fontsize=8, color=BIASED_COLOR)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Summary metrics
        avg_biased_gap = diff_df["biased_gap"].abs().mean() * 100
        avg_fair_gap = diff_df["fair_gap"].abs().mean() * 100
        avg_reduction = diff_df["gap_reduction"].mean() * 100
        pairs_improved = (diff_df["gap_reduction"] > 0).sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Gap — Biased Model", f"{avg_biased_gap:.2f}pp")
        with col2:
            st.metric("Avg Gap — Fair Model", f"{avg_fair_gap:.2f}pp")
        with col3:
            st.metric("Average Gap Reduction", f"{avg_reduction:.2f}pp")
        with col4:
            st.metric("Pairs With Improved Equity", f"{pairs_improved}/10")

        # ── INDIVIDUAL PAIR EXPLORER ────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Explore Each Resume Pair")

        selected_pair = st.selectbox(
            "Select a resume pair to examine:",
            options=list(range(10)),
            format_func=lambda i: f"Pair {i+1}: {diff_df.iloc[i]['scenario']}"
        )

        pair_row = diff_df.iloc[selected_pair]

        # Narrative
        st.markdown(f"""
        <div style='background:#e8eaf6; border-radius:10px; padding:20px; margin:12px 0;'>
            <div style='font-size:0.8rem; letter-spacing:0.1em; color:#3949ab; font-weight:700; margin-bottom:8px;'>
                SCENARIO {pair_row['pair_id']} — {pair_row['scenario'].upper()}
            </div>
            <div style='font-size:0.95rem; line-height:1.8; color:#1a237e;'>
                {pair_row['narrative']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Resume comparison
        col1, col2 = st.columns(2)
        for col, gender, name_key, uni_key, act_key, biased_key, fair_key, color, bg in [
            (col1, "Male", "male_name", "male_university", "male_activity",
             "male_biased", "male_fair", MALE_COLOR, "#e3f2fd"),
            (col2, "Female", "female_name", "female_university", "female_activity",
             "female_biased", "female_fair", FEMALE_COLOR, "#fce4ec"),
        ]:
            with col:
                b_score = pair_row[biased_key]
                f_score = pair_row[fair_key]
                b_rec = "✅ Recommend" if b_score >= 0.5 else "❌ Below Threshold"
                f_rec = "✅ Recommend" if f_score >= 0.5 else "❌ Below Threshold"

                st.markdown(f"""
                <div style='background:{bg}; border-radius:12px; padding:20px;
                            border: 2px solid {color};'>
                    <div style='color:{color}; font-weight:700; font-size:1.1rem; margin-bottom:12px;'>
                        {gender} Candidate
                    </div>
                    <div style='background:white; border-radius:8px; padding:12px; margin-bottom:12px;
                                font-size:0.9rem; line-height:1.9;'>
                        <b>Name:</b> {pair_row[name_key]}<br>
                        <b>University:</b> {pair_row[uni_key]}<br>
                        <b>Activity:</b> {pair_row[act_key]}<br>
                        <b>All other qualifications:</b> Identical to paired candidate
                    </div>
                    <div style='display:flex; gap:8px;'>
                        <div style='flex:1; background:#ffebee; border-radius:8px; padding:10px; text-align:center;'>
                            <div style='font-size:0.7rem; color:#c62828; font-weight:700;'>BIASED MODEL</div>
                            <div style='font-size:1.6rem; font-weight:800; color:#c62828;'>{b_score:.3f}</div>
                            <div style='font-size:0.75rem;'>{b_rec}</div>
                        </div>
                        <div style='flex:1; background:#e8f5e9; border-radius:8px; padding:10px; text-align:center;'>
                            <div style='font-size:0.7rem; color:#2e7d32; font-weight:700;'>FAIR MODEL</div>
                            <div style='font-size:1.6rem; font-weight:800; color:#2e7d32;'>{f_score:.3f}</div>
                            <div style='font-size:0.75rem;'>{f_rec}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Score gap analysis for this pair
        b_gap = pair_row["biased_gap"] * 100
        f_gap = pair_row["fair_gap"] * 100
        reduction = pair_row["gap_reduction"] * 100

        st.markdown(f"""
        <div style='margin-top:16px; background:white; border-radius:10px; padding:16px;
                    border:1px solid #e0e0e0;'>
            <div style='font-weight:700; margin-bottom:8px;'>Score Gap Analysis for This Pair</div>
            <div style='display:flex; gap:16px; font-size:0.9rem;'>
                <div>🔴 <b>Biased model gap:</b> {b_gap:+.2f}pp {"(male scored higher)" if b_gap > 0 else "(female scored higher)"}</div>
                <div>🟢 <b>Fair model gap:</b> {f_gap:+.2f}pp</div>
                <div>{'✅' if reduction > 0 else '⚠️'} <b>Gap reduction:</b> {reduction:.2f}pp</div>
            </div>
            <div style='margin-top:8px; font-size:0.85rem; color:#666;'>
                🔴 HUMAN REVIEW REQUIRED — These are AI recommendations, not hiring decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # TAB 2: CUSTOM CANDIDATE EVALUATION
    # ================================================================
    with tab2:
        st.markdown("## Evaluate a Custom Candidate")
        st.markdown("""
        Enter candidate details to receive AI hiring recommendations from both models.
        All outputs require human review — the AI is a decision-support tool only.
        """)

        if not _check_models():
            return

        with st.form("candidate_form"):
            st.markdown("#### Candidate Information")
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Candidate Name", placeholder="e.g., Alex Johnson")
                years_exp = st.slider("Years of Experience", 0, 10, 2)
                education = st.selectbox("Education Level",
                    ["High School (0)", "Bachelor's (1)", "Master's (2)", "PhD (3)"])
                education_num = int(education.split("(")[1].rstrip(")"))
                prog_skill = st.slider("Programming Skill (0–100)", 0, 100, 70)
                leadership = st.slider("Leadership Score (0–100)", 0, 100, 65)

            with col2:
                communication = st.slider("Communication Score (0–100)", 0, 100, 68)
                company_tier = st.selectbox("Previous Company Tier",
                    ["1 — Startup", "2 — Mid-size", "3 — Large Corp", "4 — Top Tech (FAANG)"])
                company_tier_num = int(company_tier.split("—")[0].strip())
                projects = st.slider("Number of Notable Projects", 0, 8, 3)
                interview = st.slider("Interview Score (0–100)", 0, 100, 72)

            submitted = st.form_submit_button("🔍 Evaluate Candidate", type="primary")

        if submitted:
            if not name:
                st.error("Please enter a candidate name.")
            else:
                features = {
                    "years_experience": years_exp,
                    "education_level": education_num,
                    "programming_skill": prog_skill,
                    "leadership_score": leadership,
                    "communication_score": communication,
                    "company_tier": company_tier_num,
                    "project_experience": projects,
                    "interview_score": interview,
                }

                biased_model = st.session_state.biased_model
                fair_model = st.session_state.fair_model
                scaler = st.session_state.fair_scaler

                X = pd.DataFrame([features])
                biased_score = float(biased_model.predict_proba(X)[0][1])

                X_scaled = scaler.transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
                fair_score = float(fair_model.predict_proba(X_scaled_df)[0][1])

                b_rec = "Recommend for Interview" if biased_score >= 0.5 else "Below Threshold"
                f_rec = "Recommend for Interview" if fair_score >= 0.5 else "Below Threshold"

                # Log this evaluation
                eval_id = log_evaluation(
                    candidate_name=name,
                    candidate_features=features,
                    biased_score=biased_score,
                    fair_score=fair_score,
                    biased_recommendation=b_rec,
                    fair_recommendation=f_rec,
                )
                st.session_state.last_evaluation = eval_id

                # Results
                st.markdown(f"### Results for: {name}")

                st.markdown("""
                <div class='human-review-badge'>
                    🔴 HUMAN REVIEW REQUIRED — AI recommendation only, not a hiring decision
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    score_color = "#c62828" if biased_score < 0.5 else "#1b5e20"
                    st.markdown(f"""
                    <div style='background:#ffebee; border-radius:12px; padding:24px;
                                border:2px solid #c62828; text-align:center;'>
                        <div style='font-size:0.85rem; font-weight:700; color:#c62828; margin-bottom:8px;'>
                            🔴 BIASED BASELINE MODEL
                        </div>
                        <div style='font-size:3rem; font-weight:800; color:{score_color};'>
                            {biased_score:.3f}
                        </div>
                        <div style='font-size:0.9rem; margin-top:8px; font-weight:600;'>
                            {b_rec}
                        </div>
                        <div style='font-size:0.75rem; color:#666; margin-top:8px;'>
                            Trained on historically biased data.<br>May reflect gender patterns.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    score_color2 = "#2e7d32" if fair_score >= 0.5 else "#4e342e"
                    st.markdown(f"""
                    <div style='background:#e8f5e9; border-radius:12px; padding:24px;
                                border:2px solid #2e7d32; text-align:center;'>
                        <div style='font-size:0.85rem; font-weight:700; color:#2e7d32; margin-bottom:8px;'>
                            🟢 FAIRNESS-AWARE MODEL
                        </div>
                        <div style='font-size:3rem; font-weight:800; color:{score_color2};'>
                            {fair_score:.3f}
                        </div>
                        <div style='font-size:0.9rem; margin-top:8px; font-weight:600;'>
                            {f_rec}
                        </div>
                        <div style='font-size:0.75rem; color:#666; margin-top:8px;'>
                            Fairness-improved model.<br>Not perfectly fair — human review essential.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Recruiter override
                st.markdown("---")
                st.markdown("#### 👤 Recruiter Override")
                st.markdown("""
                As a recruiter, you have the authority to override the AI recommendation.
                Your decision will be logged in the governance audit trail.
                """)

                col1, col2, col3, col4 = st.columns(4)
                notes = st.text_input("Override notes (optional):", placeholder="Reason for override...")

                with col1:
                    if st.button("✅ Approve Candidate", use_container_width=True):
                        log_recruiter_override(eval_id, "APPROVED", notes)
                        st.success("Override logged: APPROVED")
                with col2:
                    if st.button("❌ Reject Candidate", use_container_width=True):
                        log_recruiter_override(eval_id, "REJECTED", notes)
                        st.error("Override logged: REJECTED")
                with col3:
                    if st.button("⏸️ Defer for Review", use_container_width=True):
                        log_recruiter_override(eval_id, "DEFERRED", notes)
                        st.info("Override logged: DEFERRED")
                with col4:
                    if st.button("📋 Accept AI Recommendation", use_container_width=True):
                        st.info("No override. AI recommendation accepted.")

                st.caption(f"Evaluation ID: `{eval_id}` — See Audit Log for full record.")
