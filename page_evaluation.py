"""
page_evaluation.py
------------------
Renders the Candidate Evaluation and Gender Signals Experiment page.
Fixes the vertical axis stretching bug and ensures reliable model scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import models
import resume_pairs

def render():
    st.title("🤝 Candidate Evaluation & Gender Signals Experiment")
    st.markdown("""
    This experiment demonstrates how bias manifests through **proxy variables** and implicit signals. 
    We evaluate **10 pairs of identical resumes** where the only differences are gendered phrases 
    (e.g., *'Women's Chess Club'* vs *'Chess Club'*).
    """)

    # ----------------------------------------------------------------
    # 1. 核心安全检查：如果模型未初始化，提供一键就地训练，防止数据为 0 导致报错
    # ----------------------------------------------------------------
    if not st.session_state.models_trained:
        st.warning("⚠️ Models are not initialized yet. Please train the models to view the experiment.")
        
        # 尝试寻找数据集
        csv_path = "synthetic_hiring_data.csv"
        if os.path.exists(csv_path):
            if st.button("🚀 Initialize & Train Models Right Here", type="primary"):
                with st.spinner("Injecting historical bias and training neural clusters..."):
                    df = pd.read_csv(csv_path)
                    X_train, X_test, y_train, y_test, g_train, g_test = models.prepare_data(df)
                    
                    # 训练并存入全局状态
                    st.session_state.biased_model = models.train_biased_model(X_train, y_train)
                    fair_model, fair_scaler = models.train_fairness_aware_model(X_train, y_train, g_train)
                    st.session_state.fair_model = fair_model
                    st.session_state.fair_scaler = fair_scaler
                    st.session_state.dataset = df
                    st.session_state.models_trained = True
                    st.rerun()
            return
        else:
            st.error(f"Missing '{csv_path}' in project directory. Please check your GitHub repository.")
            return

    # ----------------------------------------------------------------
    # 2. 提取数据并计算得分差距
    # ----------------------------------------------------------------
    with st.spinner("Evaluating resume standard pairs..."):
        pairs_df = resume_pairs.get_all_pairs_dataframe()
        differentials = models.score_resume_pairs(
            pairs_df,
            st.session_state.biased_model,
            st.session_state.fair_model,
            st.session_state.fair_scaler
        )

    # ----------------------------------------------------------------
    # 3. 渲染顶部的看板指标 (Metrics Dashboard)
    # ----------------------------------------------------------------
    # 将概率差距转换为百分点显示 (pp)
    avg_biased_gap = differentials["biased_score_gap"].mean() * 100
    avg_fair_gap = differentials["fair_score_gap"].mean() * 100
    avg_reduction = differentials["gap_reduction"].mean() * 100
    improved_pairs = sum(differentials["gap_reduction"] > 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Gap — Biased Model", f"{avg_biased_gap:+.2f}pp")
    m2.metric("Avg Gap — Fair Model", f"{avg_fair_gap:+.2f}pp")
    m3.metric("Average Gap Reduction", f"{avg_reduction:.2f}pp", delta=f"{avg_reduction:.2f}pp" if avg_reduction > 0 else None)
    m4.metric("Pairs With Improved Equity", f"{improved_pairs}/10")

    st.divider()

    # ----------------------------------------------------------------
    # 4. 绘制柱状图（完美修复坐标轴动态拉伸问题）
    # ----------------------------------------------------------------
    st.subheader("Visualizing the Equity Gaps")
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(differentials))
    width = 0.35

    # 统一换算为百分点数据进行绘图
    biased_plot_data = differentials["biased_score_gap"] * 100
    fair_plot_data = differentials["fair_score_gap"] * 100

    rects1 = ax.bar(x - width/2, biased_plot_data, width, label="Biased Model Gap", color="#D9534F")
    rects2 = ax.bar(x + width/2, fair_plot_data, width, label="Fairness-Aware Model Gap", color="#5CB85C")

    # 设定平滑的纵轴上下限，避免极小值时视觉缩放产生误导
    max_val = max(biased_plot_data.max(), fair_plot_data.max(), 5.0)
    min_val = min(biased_plot_data.min(), fair_plot_data.min(), -5.0)
    ax.set_ylim(min_val * 1.3, max_val * 1.3)

    ax.set_ylabel("Score Gap: Male - Female (Percentage Points)", fontsize=10)
    ax.set_title("Gender Score Gaps — Identical Qualifications, Only Gendered Signals Differ\nPositive = Male scored higher | Closer to 0 = More equitable", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pair {i+1}" for i in range(len(differentials))], rotation=25, ha="right")
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)

    # =========================================================================
    # 【完美解决拉伸 Bug】：只有当存在真实的偏见差距时，才采用自适应高度进行箭头标注
    # =========================================================================
    if biased_plot_data.max() > 0.01:
        max_idx = biased_plot_data.idxmax()
        peak_y = biased_plot_data.max()
        # 动态偏移量：设为当前纵轴总视窗高度的 15%，绝不产生过大硬编码越界
        dynamic_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15
        
        ax.annotate(
            f"Largest gap:\n{peak_y:.2f}pp",
            xy=(max_idx - width/2, peak_y),
            xytext=(max_idx - width/2, peak_y + dynamic_offset),
            arrowprops=dict(facecolor="#D9534F", edgecolor="#D9534F", arrowstyle="->", connectionstyle="arc3,rad=.1"),
            color="#D9534F",
            fontweight="bold",
            ha="center",
            va="bottom"
        )
    else:
        # 如果差距为 0，展示一个温和的中央中性提示，不画箭头
        ax.text(4.5, (ax.get_ylim()[1])*0.7, "No noticeable gaps detected. Please ensure models are trained with biased data.", 
                color="#777777", style="italic", ha="center", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    
    # 输出到 Streamlit
    st.pyplot(fig)

    # ----------------------------------------------------------------
    # 5. 底部提供交互式简历对查看器 (Interactive Resumes Inspector)
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Deep-Dive: Inspecting Resume Pairs")
    
    pair_selection = st.selectbox("Select a pair to audit details:", [f"Resume Pair {i+1}: {resume_pairs.RESUME_PAIRS[i]['scenario']}" for i in range(10)])
    selected_idx = int(pair_selection.split(":")[0].replace("Resume Pair ", "")) - 1
    
    raw_pair = resume_pairs.RESUME_PAIRS[selected_idx]
    diff_row = differentials.iloc[selected_idx]
    
    st.info(f"**Scenario Context:** {raw_pair['narrative']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"##### 👨‍💻 Male Candidate: {raw_pair['male']['name']}")
        st.caption(f"**Background:** {raw_pair['male']['university']} | {raw_pair['male']['activity']}")
        st.metric("Biased Prediction Score", f"{diff_row['male_biased_score']*100:.1f}%")
        st.metric("Fairness Model Score", f"{diff_row['male_fair_score']*100:.1f}%")
        
    with col2:
        st.markdown(f"##### 👩‍💻 Female Candidate: {raw_pair['female']['name']}")
        st.caption(f"**Background:** {raw_pair['female']['university']} | {raw_pair['female']['activity']}")
        st.metric("Biased Prediction Score", f"{diff_row['female_biased_score']*100:.1f}%", 
                  delta=f"{(diff_row['female_biased_score'] - diff_row['male_biased_score'])*100:.1f}% Bias Penalty" if diff_row['biased_score_gap'] != 0 else None,
                  delta_color="inverse")
        st.metric("Fairness Model Score", f"{diff_row['female_fair_score']*100:.1f}%")
