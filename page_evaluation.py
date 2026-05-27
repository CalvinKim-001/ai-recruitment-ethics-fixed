"""
page_evaluation.py
------------------
Interactive Workspace integrating Single Candidate Auditing & Resume Pairs Experimentation.
Restores candidate name inputs, human-in-the-loop overrides, and introduces instant preset loading buttons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import models
import resume_pairs

def render():
    st.title("⚖️ Candidate Evaluation Workspace")
    st.markdown("Explore individual screening assessments or evaluate paired compliance experiments.")

    # Model training validation gate
    if not st.session_state.models_trained:
        st.warning("⚙️ AI Models are not initialized yet. Please navigate to the **Fairness Audit Dashboard** to trigger training routines first.")
        return

    # Initialize form preset variables in session state if not present
    if "preset_name" not in st.session_state:
        st.session_state.preset_name = "Alex Morgan"
    if "preset_exp" not in st.session_state:
        st.session_state.preset_exp = 3
    if "preset_edu" not in st.session_state:
        st.session_state.preset_edu = "Bachelor's"
    if "preset_tier" not in st.session_state:
        st.session_state.preset_tier = "Mid-size"
    if "preset_prog" not in st.session_state:
        st.session_state.preset_prog = 75
    if "preset_lead" not in st.session_state:
        st.session_state.preset_lead = 60
    if "preset_comm" not in st.session_state:
        st.session_state.preset_comm = 65
    if "preset_proj" not in st.session_state:
        st.session_state.preset_proj = 2
    if "preset_inter" not in st.session_state:
        st.session_state.preset_inter = 70
    if "preset_signal" not in st.session_state:
        st.session_state.preset_signal = "Standard Profile"

    # Use Streamlit Tabs to keep interactive features and experiments organized
    tab1, tab2 = st.tabs(["🎯 Interactive Candidate Evaluator", "📊 Gender Signals Pair Experiment"])

    # =========================================================================
    # TAB 1: Interactive Candidate Evaluator (With Manual Clicking Presets)
    # =========================================================================
    with tab1:
        st.subheader("Evaluate Custom Candidate Specifications")
        st.markdown("Click one of the quick-load presets below or manually adjust the parameters to test the models.")
        
        # 【기능 보완】: 사용자가 직접 클릭하여 프로필을 즉시 불러올 수 있는 수동 클릭 옵션 버튼
        st.markdown("##### 💡 Quick Load Candidate Presets (인공 클릭 옵션)")
        p_col1, p_col2, p_col3 = st.columns(3)
        
        if p_col1.button("👨 Load Qualified Male Profile (James)"):
            st.session_state.preset_name = "James Whitfield"
            st.session_state.preset_exp = 2
            st.session_state.preset_edu = "Bachelor's"
            st.session_state.preset_tier = "Mid-size"
            st.session_state.preset_prog = 88
            st.session_state.preset_lead = 82
            st.session_state.preset_comm = 78
            st.session_state.preset_proj = 4
            st.session_state.preset_inter = 85
            st.session_state.preset_signal = "Standard Profile"
            st.rerun()
            
        if p_col2.button("👩 Load Paired Female Profile with 'Women's' Signal (Claire)"):
            st.session_state.preset_name = "Claire Whitfield"
            st.session_state.preset_exp = 2
            st.session_state.preset_edu = "Bachelor's"
            st.session_state.preset_tier = "Mid-size"
            st.session_state.preset_prog = 88
            st.session_state.preset_lead = 82
            st.session_state.preset_comm = 78
            st.session_state.preset_proj = 4
            st.session_state.preset_inter = 85
            st.session_state.preset_signal = "Features Gendered Phrases (e.g., Women's Club)"
            st.rerun()
            
        if p_col3.button("🔄 Reset to Default Blank Profile"):
            st.session_state.preset_name = "Alex Morgan"
            st.session_state.preset_exp = 3
            st.session_state.preset_edu = "Bachelor's"
            st.session_state.preset_tier = "Mid-size"
            st.session_state.preset_prog = 75
            st.session_state.preset_lead = 60
            st.session_state.preset_comm = 65
            st.session_state.preset_proj = 2
            st.session_state.preset_inter = 70
            st.session_state.preset_signal = "Standard Profile"
            st.rerun()

        st.divider()

        with st.form("interactive_evaluator_form"):
            # 후보자 이름 입력 상자 복원
            candidate_name = st.text_input("Candidate Name", value=st.session_state.preset_name)
            st.divider()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                exp = st.slider("Years of Industry Experience", 0, 10, value=st.session_state.preset_exp)
                
                edu_options = ["High School", "Bachelor's", "Master's", "PhD"]
                edu = st.selectbox("Completed Education Level", edu_options, index=edu_options.index(st.session_state.preset_edu))
                
                tier_options = ["Startup", "Mid-size", "Large Corp", "Top Tech (FAANG)"]
                tier = st.selectbox("Previous Company Profile Tier", tier_options, index=tier_options.index(st.session_state.preset_tier))
            with c2:
                prog = st.slider("Programming Mastery Score", 0, 100, value=st.session_state.preset_prog)
                lead = st.slider("Leadership Aptitude Vector", 0, 100, value=st.session_state.preset_lead)
                comm = st.slider("Communication Articulation Index", 0, 100, value=st.session_state.preset_comm)
            with c3:
                proj = st.slider("Completed Open Source Projects", 0, 5, value=st.session_state.preset_proj)
                inter = st.slider("Live Technical Interview Evaluation", 0, 100, value=st.session_state.preset_inter)
                
                sig_options = ["Standard Profile", "Features Gendered Phrases (e.g., Women's Club)"]
                signal_type = st.radio("Resume Phrasing Signal Pattern", sig_options, index=sig_options.index(st.session_state.preset_signal))
            
            st.divider()
            # 인사담당자 관리자 권한 Override 항목 및 거버넌스 피드백 입력란 복원
            st.markdown("##### 👥 Human-in-the-Loop Governance Override")
            hr_override = st.selectbox(
                "Recruiter Override AI Algorithmic Recommendation?", 
                ["No Override - Follow AI Advice", "Force Approve / Recommend for Interview", "Force Reject / Decline Profile"]
            )
            hr_feedback = st.text_area(
                "HR Governance Feedback & Auditing Notes", 
                placeholder="Provide qualitative justification for manual override, bias mitigation tracking or compliance record logs..."
            )

            submitted = st.form_submit_button("Run Algorithmic Auditing & Log Entry", type="primary")

        if submitted:
            edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
            tier_map = {"Startup": 1, "Mid-size": 2, "Large Corp": 3, "Top Tech (FAANG)": 4}
            sig_val = 0.15 if signal_type == "Features Gendered Phrases (e.g., Women's Club)" else 0.85

            candidate_payload = {
                "years_experience": exp,
                "education_level": edu_map[edu],
                "programming_skill": prog,
                "leadership_score": lead,
                "communication_score": comm,
                "company_tier": tier_map[tier],
                "project_experience": proj,
                "interview_score": inter,
                "resume_gender_signal": sig_val
            }
            
            evaluation_res = models.score_candidate(
                candidate_payload,
                st.session_state.biased_model,
                st.session_state.fair_model,
                st.session_state.fair_scaler
            )
            
            # =========================================================================
            # 【这里是核心修复位置】：将选择框翻译成表格需要的 Approved / Rejected 状态
            # 并同时写入大写首字母键名与小写键名，实现 100% 数据流多态同步联动
            # =========================================================================
            decision_mapped = "No Override"
            if "Approve" in hr_override:
                decision_mapped = "Approved"
            elif "Reject" in hr_override or "Decline" in hr_override:
                decision_mapped = "Rejected"

            audit_entry = {
                # 1. 注入大写键名（精确同步 page_audit.py 页面表格与计数卡片）
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Candidate": candidate_name,
                "Biased Score": f"{evaluation_res['biased_score']*100:.1f}%",
                "Fair Score": f"{evaluation_res['fair_score']*100:.1f}%",
                "Decision": decision_mapped,
                "Notes": hr_feedback if hr_feedback.strip() else "Audit synced.",
                
                # 2. 原封不动保留小写备份，防止破坏系统任何其他辅助组件
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidate_name": candidate_name,
                "candidate": candidate_name,
                "biased_score": evaluation_res['biased_score'],
                "fair_score": evaluation_res['fair_score'],
                "override_status": hr_override,
                "decision": decision_mapped.lower(),
                "feedback": hr_feedback if hr_feedback.strip() else "Audit synced.",
                "notes": hr_feedback if hr_feedback.strip() else "Audit synced."
            }
            st.session_state.audit_entries.append(audit_entry)
            
            st.markdown("### 🔍 Evaluation Metrics Analysis")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Biased System Recommendation Score", f"{evaluation_res['biased_score']*100:.1f}%")
                if evaluation_res['biased_score'] >= 0.5:
                    st.success(f"Outcome: {evaluation_res['biased_recommendation']}")
                else:
                    st.error(f"Outcome: {evaluation_res['biased_recommendation']}")
            with res_col2:
                st.metric("Fairness-Improved Recommendation Score", f"{evaluation_res['fair_score']*100:.1f}%")
                if evaluation_res['fair_score'] >= 0.5:
                    st.success(f"Outcome: {evaluation_res['fair_recommendation']}")
                else:
                    st.error(f"Outcome: {evaluation_res['fair_recommendation']}")
                    
            st.success("📝 Candidate evaluated and governance audit log entry saved successfully! View the records on the Governance page.")

    # =========================================================================
    # TAB 2: Gender Signals Pair Experiment (With Enhanced Descriptive Text)
    # =========================================================================
    with tab2:
        st.subheader("Gender Signals Audit Dashboard (10 Matched Profiles)")
        st.markdown("Identical academic and work backgrounds. The only variance lies within proxy linguistic flags.")

        pairs_df = resume_pairs.get_all_pairs_dataframe()
        differentials = models.score_resume_pairs(
            pairs_df,
            st.session_state.biased_
