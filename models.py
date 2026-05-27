"""
models.py
---------
Defines and trains both the baseline biased and fairness-aware recruitment models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLUMNS = [
    "years_experience",
    "education_level",
    "programming_skill",
    "leadership_score",
    "communication_score",
    "company_tier",
    "project_experience",
    "interview_score",
    "resume_gender_signal",
]

TARGET_COLUMN = "hired"
SENSITIVE_COLUMN = "gender"


def prepare_data(df: pd.DataFrame):
    """
    Split clean dataset into train/test components natively.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    gender = df[SENSITIVE_COLUMN]

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, gender,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, g_train, g_test


def train_biased_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train pipeline using baseline imbalanced historic patterns."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_fairness_aware_model(X_train: pd.DataFrame, y_train: pd.Series, gender_train: pd.Series) -> tuple:
    """Train a fairness-mitigated system via Fairlearn Equalized Odds constraints."""
    base_classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    constraint = EqualizedOdds()

    fairness_model = ExponentiatedGradient(
        estimator=base_classifier,
        constraints=constraint,
        eps=0.01
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

    fairness_model.fit(
        X_scaled_df,
        y_train,
        sensitive_features=gender_train.values
    )

    class FairlearnProbWrapper:
        def __init__(self, model):
            self.model = model
            self.classes_ = np.array([0, 1])

        def predict(self, X):
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

        def predict_proba(self, X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
            proba = np.zeros((X.shape[0], 2))
            for predictor, weight in zip(self.model.predictors_, self.model.weights_):
                proba += weight * predictor.predict_proba(X)
            return proba

    wrapped_model = FairlearnProbWrapper(fairness_model)
    return wrapped_model, scaler


def score_candidate(candidate_features: dict, biased_model, fairness_model, scaler) -> dict:
    """Score a single candidate interactively via both active models."""
    X = pd.DataFrame([{col: candidate_features.get(col, 0) for col in FEATURE_COLUMNS}])

    biased_prob = biased_model.predict_proba(X)[0][1]
    biased_rec = "Recommend for Interview" if biased_prob >= 0.5 else "Does Not Meet Threshold"

    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    fair_prob = fairness_model.predict_proba(X_scaled_df)[0][1]
    fair_rec = "Recommend for Interview" if fair_prob >= 0.5 else "Does Not Meet Threshold"

    return {
        "biased_score": round(float(biased_prob), 4),
        "biased_recommendation": biased_rec,
        "fair_score": round(float(fair_prob), 4),
        "fair_recommendation": fair_rec,
        "human_review_required": True,
    }


def score_resume_pairs(pairs_df: pd.DataFrame, biased_model, fairness_model, scaler) -> pd.DataFrame:
    """Process paired experimental configurations."""
    results = []
    for _, row in pairs_df.iterrows():
        features = {col: row[col] for col in FEATURE_COLUMNS}
        X = pd.DataFrame([features])
        biased_prob = biased_model.predict_proba(X)[0][1]

        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
        fair_prob = fairness_model.predict_proba(X_scaled_df)[0][1]

        results.append({
            "pair_id": row["pair_id"],
            "name": row["name"],
            "gender": row["gender"],
            "scenario": row["scenario"],
            "biased_score": round(float(biased_prob), 4),
            "fair_score": round(float(fair_prob), 4),
        })

    results_df = pd.DataFrame(results)
    differentials = []
    for pair_id in results_df["pair_id"].unique():
        pair = results_df[results_df["pair_id"] == pair_id]
        male_row = pair[pair["gender"] == "Male"].iloc[0]
        female_row = pair[pair["gender"] == "Female"].iloc[0]

        biased_diff = round(male_row["biased_score"] - female_row["biased_score"], 4)
        fair_diff = round(male_row["fair_score"] - female_row["fair_score"], 4)

        differentials.append({
            "pair_id": pair_id,
            "scenario": male_row["scenario"],
            "male_biased_score": male_row["biased_score"],
            "female_biased_score": female_row["biased_score"],
            "biased_score_gap": biased_diff,
            "male_fair_score": male_row["fair_score"],
            "female_fair_score": female_row["fair_score"],
            "fair_score_gap": fair_diff,
            "gap_reduction": round(abs(biased_diff) - abs(fair_diff), 4),
        })
    return pd.DataFrame(differentials)
