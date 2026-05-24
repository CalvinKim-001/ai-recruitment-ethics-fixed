"""
data_generator.py
-----------------
Generates synthetic hiring data for the AI Recruitment Bias project.

ETHICAL NOTE:
This data is intentionally imbalanced. We simulate historical human bias
not by explicit gender discrimination, but through PROXY VARIABLES (like 
resume language, clubs, and college names). 

This teaches the core lesson of the Amazon AI case: algorithms will find
and penalize female-associated patterns even if the "Gender" column is hidden.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_hiring_dataset(n_candidates: int = 1500) -> pd.DataFrame:
    
    # 1. Assign gender (72% Male, 28% Female)
    gender = np.random.choice(
        ["Male", "Female"],
        size=n_candidates,
        p=[0.72, 0.28]
    )

    # 2. Objective Qualifications (Similar across genders)
    years_experience = np.random.randint(0, 8, size=n_candidates)
    education_level = np.random.choice([0, 1, 2, 3], size=n_candidates, p=[0.05, 0.60, 0.30, 0.05])
    programming_skill = np.clip(np.random.normal(loc=65, scale=18, size=n_candidates), 0, 100).astype(int)
    leadership_score = np.clip(np.random.normal(loc=60, scale=15, size=n_candidates), 0, 100).astype(int)
    communication_score = np.clip(np.random.normal(loc=62, scale=14, size=n_candidates), 0, 100).astype(int)
    company_tier = np.random.choice([1, 2, 3, 4], size=n_candidates, p=[0.3, 0.4, 0.2, 0.1])
    project_experience = np.random.randint(0, 6, size=n_candidates)
    interview_score = np.clip(np.random.normal(loc=63, scale=16, size=n_candidates), 0, 100).astype(int)

    # ----------------------------------------------------------------
    # 核心修复：引入“简历性别信号” (Proxy Variable)
    # ----------------------------------------------------------------
    # 0 = 包含大量女性暗示词 (如 Women's Club, Wellesley College)
    # 1 = 包含大量男性/中性暗示词
    # 这是一个隐藏特征，AI 将通过它学会隐性歧视
    resume_gender_signal = np.where(
        gender == "Male",
        np.clip(np.random.normal(loc=0.8, scale=0.15, size=n_candidates), 0, 1),
        np.clip(np.random.normal(loc=0.2, scale=0.15, size=n_candidates), 0, 1)
    )

    # 3. 历史偏见录用概率计算
    # 历史上的 HR 看到了偏女性的简历用词，潜意识里给出了较低的评分
    base_hire_prob = (
        0.05 * years_experience +
        0.08 * education_level +
        0.003 * programming_skill +
        0.002 * leadership_score +
        0.001 * communication_score +
        0.04 * company_tier +
        0.02 * project_experience +
        0.003 * interview_score +
        0.30 * resume_gender_signal  # <--- 巨大的历史偏见权重在这里！
    )

    # Normalize to a 0–1 probability range
    base_hire_prob = (base_hire_prob - base_hire_prob.min()) / (base_hire_prob.max() - base_hire_prob.min())
    
    # Generate labels
    hired = np.random.binomial(1, base_hire_prob)

    # 4. Assemble DataFrame
    df = pd.DataFrame({
        "candidate_id": [f"C{str(i).zfill(4)}" for i in range(1, n_candidates + 1)],
        "gender": gender,
        "resume_gender_signal": round(pd.Series(resume_gender_signal), 3), # 新增特征
        "years_experience": years_experience,
        "education_level": education_level,
        "programming_skill": programming_skill,
        "leadership_score": leadership_score,
        "communication_score": communication_score,
        "company_tier": company_tier,
        "project_experience": project_experience,
        "interview_score": interview_score,
        "hired": hired 
    })

    return df

# =================================================================
# 请把以下三个工具函数补充粘贴到 data_generator.py 的最底部
# =================================================================

def get_education_label(level: int) -> str:
    """Convert numeric education level to human-readable label."""
    mapping = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
    return mapping.get(level, "Unknown")

def get_company_tier_label(tier: int) -> str:
    """Convert numeric company tier to human-readable label."""
    mapping = {1: "Startup", 2: "Mid-size", 3: "Large Corp", 4: "Top Tech (FAANG)"}
    return mapping.get(tier, "Unknown")

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the dataset's gender distribution and hiring rates.
    This is used to demonstrate the imbalance before any model is trained.
    """
    summary = {}
    for gender in ["Male", "Female"]:
        subset = df[df["gender"] == gender]
        summary[gender] = {
            "count": len(subset),
            "pct_of_dataset": round(len(subset) / len(df) * 100, 1),
            "hire_rate": round(subset["hired"].mean() * 100, 1),
            "avg_programming_skill": round(subset["programming_skill"].mean(), 1),
            "avg_interview_score": round(subset["interview_score"].mean(), 1),
        }
    return summary
