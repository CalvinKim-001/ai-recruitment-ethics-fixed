"""
data_generator.py
-----------------
Generates synthetic hiring data for the AI Recruitment Bias project.

ETHICAL NOTE:
This data is intentionally imbalanced — male candidates are given higher
historical hiring probabilities. This is NOT because we believe men are
better candidates. It simulates how real-world biased historical data
(like Amazon's 10 years of male-dominated hiring records) causes AI systems
to learn discriminatory patterns. The bias is in the data, not in reality.

This is a core lesson of the project: garbage in, garbage out.
If historical data reflects discrimination, the AI will reproduce it.
"""

import numpy as np
import pandas as pd

# Fix the random seed so results are reproducible every time the code runs.
# This is important for academic projects — your professor should see the same
# results you do.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_hiring_dataset(n_candidates: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic dataset of junior software engineer candidates.

    The dataset intentionally contains historical gender imbalance:
    - ~70% of candidates are male (reflecting tech industry historical norms)
    - Male candidates have a higher baseline probability of being labeled
      as 'hired' in the historical data

    This simulates exactly the kind of biased training data that caused
    Amazon's recruitment AI to penalize women's resumes.

    Parameters:
        n_candidates: Number of synthetic candidates to generate (default 1000)

    Returns:
        A pandas DataFrame with one row per candidate
    """

    # ----------------------------------------------------------------
    # STEP 1: Assign gender with realistic tech-industry imbalance
    # ----------------------------------------------------------------
    # Historically, ~70-75% of tech hires were male (Stack Overflow surveys,
    # Bureau of Labor Statistics data). We replicate this imbalance here.
    # ETHICAL NOTE: This skew is the ROOT CAUSE of the bias the AI will learn.
    gender = np.random.choice(
        ["Male", "Female"],
        size=n_candidates,
        p=[0.72, 0.28]  # 72% male, 28% female — intentional historical imbalance
    )

    # ----------------------------------------------------------------
    # STEP 2: Generate candidate qualifications
    # ----------------------------------------------------------------
    # We deliberately make qualifications SIMILAR across genders.
    # This is important: the AI's eventual bias will NOT be because
    # women are less qualified in our data — it will be purely because
    # male patterns dominate the training data.

    years_experience = np.random.randint(0, 8, size=n_candidates)

    education_level = np.random.choice(
        [0, 1, 2, 3],  # 0=High School, 1=Bachelor's, 2=Master's, 3=PhD
        size=n_candidates,
        p=[0.05, 0.60, 0.30, 0.05]
    )

    # Technical skill: 0–100. Similar distributions across genders.
    programming_skill = np.clip(
        np.random.normal(loc=65, scale=18, size=n_candidates), 0, 100
    ).astype(int)

    leadership_score = np.clip(
        np.random.normal(loc=60, scale=15, size=n_candidates), 0, 100
    ).astype(int)

    communication_score = np.clip(
        np.random.normal(loc=62, scale=14, size=n_candidates), 0, 100
    ).astype(int)

    # Previous company tier: 1 (startup) to 4 (FAANG-level)
    company_tier = np.random.choice([1, 2, 3, 4], size=n_candidates, p=[0.3, 0.4, 0.2, 0.1])

    # Number of notable projects
    project_experience = np.random.randint(0, 6, size=n_candidates)

    # Interview performance score: 0–100
    interview_score = np.clip(
        np.random.normal(loc=63, scale=16, size=n_candidates), 0, 100
    ).astype(int)

    # ----------------------------------------------------------------
    # STEP 3: Generate biased historical hiring outcomes
    # ----------------------------------------------------------------
    # This is the most ethically significant step.
    # We calculate a "hire probability" that is artificially inflated
    # for male candidates — simulating decades of biased hiring decisions
    # made by human recruiters before any AI was involved.
    #
    # ETHICAL NOTE: The discrimination starts HERE, with human decisions,
    # before the AI ever touches the data. The AI then LEARNS from these
    # biased human decisions and replicates them at scale.

    base_hire_prob = (
        0.05 * years_experience +
        0.08 * education_level +
        0.003 * programming_skill +
        0.002 * leadership_score +
        0.001 * communication_score +
        0.04 * company_tier +
        0.02 * project_experience +
        0.003 * interview_score
    )

    # Normalize to a 0–1 probability range
    base_hire_prob = (base_hire_prob - base_hire_prob.min()) / (
        base_hire_prob.max() - base_hire_prob.min()
    )

    # Apply the gender bias: male candidates get a +15% hiring boost
    # This represents historical human bias baked into the training labels.
    gender_bias_boost = np.where(gender == "Male", 0.15, 0.0)
    biased_hire_prob = np.clip(base_hire_prob + gender_bias_boost, 0, 1)

    # Convert probabilities to binary outcomes (hired = 1, rejected = 0)
    hired = np.random.binomial(1, biased_hire_prob)

    # ----------------------------------------------------------------
    # STEP 4: Assemble the DataFrame
    # ----------------------------------------------------------------
    df = pd.DataFrame({
        "candidate_id": [f"C{str(i).zfill(4)}" for i in range(1, n_candidates + 1)],
        "gender": gender,
        "years_experience": years_experience,
        "education_level": education_level,
        "programming_skill": programming_skill,
        "leadership_score": leadership_score,
        "communication_score": communication_score,
        "company_tier": company_tier,
        "project_experience": project_experience,
        "interview_score": interview_score,
        "hired": hired  # 1 = historically hired, 0 = historically rejected
    })

    return df


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


if __name__ == "__main__":
    # Quick test: generate and preview the dataset
    df = generate_hiring_dataset(1000)
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nGender distribution and hiring rates:")
    summary = get_dataset_summary(df)
    for gender, stats in summary.items():
        print(f"\n  {gender}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
