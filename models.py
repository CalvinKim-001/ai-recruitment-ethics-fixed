"""
models.py
---------
Defines and trains two recruitment models:

1. BIASED BASELINE MODEL
   Trained on historically imbalanced data WITHOUT any fairness correction.
   This model will learn to associate male patterns with successful hires —
   not because we programmed discrimination in, but because the training
   data reflects historical discrimination made by humans.

2. FAIRNESS-AWARE MODEL
   Trained using sample reweighting and balanced data to reduce measurable
   gender discrimination. This model is "fairness-improved," not "perfect."
   Fairness tradeoffs still exist — we are honest about this.

ETHICAL NOTE:
Neither model is an autonomous hiring authority.
Both models produce RECOMMENDATION SCORES that require human review.
The AI assists the recruiter; it does not replace human judgment.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import warnings
warnings.filterwarnings("ignore")

# Features used for model training.
# IMPORTANT: Gender is INCLUDED as a training feature in the biased model
# to show that even explicit demographic info enters the model through the data labels.
# In the fairness-aware model, gender is used only as a fairness constraint, not a feature.
FEATURE_COLUMNS = [
    "years_experience",
    "education_level",
    "programming_skill",
    "leadership_score",
    "communication_score",
    "company_tier",
    "project_experience",
    "interview_score",
]

TARGET_COLUMN = "hired"
SENSITIVE_COLUMN = "gender"


def prepare_data(df: pd.DataFrame):
    """
    Split dataset into train/test sets and extract features, labels, and
    sensitive attributes.

    Returns:
        X_train, X_test, y_train, y_test, gender_train, gender_test
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    gender = df[SENSITIVE_COLUMN]

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, gender,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class balance in split
    )

    return X_train, X_test, y_train, y_test, g_train, g_test


def train_biased_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train the biased baseline recruitment model.

    Uses Gradient Boosting — a powerful and common real-world model choice.
    Trained on raw imbalanced data with NO fairness corrections.

    ETHICAL NOTE:
    This model will learn that "male-associated patterns" predict hiring success
    because the training labels were generated from biased historical decisions.
    The model is not programmed to discriminate — it LEARNS to discriminate
    from the data. This is the central lesson of the Amazon case.

    Returns:
        A trained sklearn Pipeline (scaler + classifier)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Normalize features
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_fairness_aware_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    gender_train: pd.Series
) -> object:
    """
    Train a fairness-aware recruitment model using Fairlearn's
    ExponentiatedGradient reduction with Equalized Odds constraint.
    """
    # Base classifier: Logistic Regression is ideal here because it's
    # interpretable and works well with Fairlearn's reduction framework.
    base_classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )

    # Fairness constraint: Equalized Odds
    constraint = EqualizedOdds()

    # ExponentiatedGradient finds the best model that satisfies the constraint
    fairness_model = ExponentiatedGradient(
        estimator=base_classifier,
        constraints=constraint,
        eps=0.01,
        sample_weight_name="sample_weight" # 增加参数以提高兼容性
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

    fairness_model.fit(
        X_scaled_df,
        y_train,
        sensitive_features=gender_train.values
    )

    # ==========================================
    # 核心修复部分：创建一个包装器来手动计算 predict_proba
    # 因为较新版本的 Fairlearn 默认不直接暴露这个方法
    # ==========================================
    class FairlearnProbWrapper:
        def __init__(self, model):
            self.model = model
            self.classes_ = np.array([0, 1])

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            # 将 DataFrame 转换为 numpy 数组以防兼容性问题
            X_array = X.values if hasattr(X, 'values') else X

            # ExponentiatedGradient 内部由多个子模型（predictors_）和对应的权重（weights_）组成
            # 我们通过对所有子模型的概率结果进行加权平均，来得出最终的概率分数
            proba = np.zeros((X_array.shape[0], 2))
            for predictor, weight in zip(self.model.predictors_, self.model.weights_):
                proba += weight * predictor.predict_proba(X_array)
            return proba

    wrapped_model = FairlearnProbWrapper(fairness_model)

    return wrapped_model, scaler


def score_candidate(candidate_features: dict, biased_model, fairness_model, scaler) -> dict:
    """
    Score a single candidate using both models.
    """
    # Convert candidate dict to DataFrame row
    X = pd.DataFrame([{col: candidate_features.get(col, 0) for col in FEATURE_COLUMNS}])

    # Biased model prediction
    biased_prob = biased_model.predict_proba(X)[0][1]
    biased_rec = "Recommend for Interview" if biased_prob >= 0.5 else "Does Not Meet Threshold"

    # Fairness-aware model prediction
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    fair_prob = fairness_model.predict_proba(X_scaled_df)[0][1]
    fair_rec = "Recommend for Interview" if fair_prob >= 0.5 else "Does Not Meet Threshold"

    return {
        "biased_score": round(float(biased_prob), 4),
        "biased_recommendation": biased_rec,
        "fair_score": round(float(fair_prob), 4),
        "fair_recommendation": fair_rec,
        "human_review_required": True,  # ALWAYS — AI assists, never decides
    }


def score_resume_pairs(pairs_df: pd.DataFrame, biased_model, fairness_model, scaler) -> pd.DataFrame:
    """
    Score all matched resume pairs and return results with score differentials.
    This is the core of the gender signals experiment.
    """
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

    # Calculate score differentials within each pair
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
            "biased_score_gap": biased_diff,  # Positive = male scored higher
            "male_fair_score": male_row["fair_score"],
            "female_fair_score": female_row["fair_score"],
            "fair_score_gap": fair_diff,       # Positive = male scored higher
            "gap_reduction": round(abs(biased_diff) - abs(fair_diff), 4),
        })

    return pd.DataFrame(differentials)


def evaluate_model(model, X_test, y_test, model_name="Model", scaler=None):
    """
    Print classification metrics for a trained model.
    """
    if scaler is not None:
        X_scaled = scaler.transform(X_test)
        X_input = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    else:
        X_input = X_test

    y_pred = model.predict(X_input)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Hired"]))

    return y_pred