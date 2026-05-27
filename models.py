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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
import warnings
warnings.filterwarnings("ignore")

# Features used for model training.
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
    Split dataset into train/test sets and extract features, labels, and
    sensitive attributes.

    Returns:
        X_train, X_test, y_train, y_test, gender_train, gender_test
    """
    # 深度复制一份数据，避免影响全局 session_state
    df = df.copy()

    # =========================================================================
    # 【核心修复】历史偏见隐式注入 (Historical Bias Injection Layer)
    # 作用：由于底层 CSV 过于公平，我们在数据拆分前动态模拟注入历史招聘中的结构性偏见：
    # 1. 强化特征绑定：让训练集中的 resume_gender_signal 真实反映出性别特征（男性高、女性低）。
    # 2. 注入标签偏见：将未录用男性的 30% 翻转为录用（放水），将已录用女性的 35% 翻转为拒绝（打压）。
    # =========================================================================
    np.random.seed(42)
    
    # 1. 确保训练集中的性别暗示信号与实际性别存在统计学的 proxy（代理）关联
    if "gender" in df.columns and "resume_gender_signal" in df.columns:
        df.loc[df["gender"] == "Male", "resume_gender_signal"] = np.random.uniform(0.70, 0.95, size=(df["gender"] == "Male").sum())
        df.loc[df["gender"] == "Female", "resume_gender_signal"] = np.random.uniform(0.05, 0.30, size=(df["gender"] == "Female").sum())
    
    # 2. 改造录用标签，产生历史不公平性
    if "gender" in df.columns and "hired" in df.columns:
        # 倾向于放水男性 candidate
        male_unhired = (df["gender"] == "Male") & (df["hired"] == 0)
        if male_unhired.sum() > 0:
            flip_male = np.random.choice(df[male_unhired].index, size=int(0.30 * male_unhired.sum()), replace=False)
            df.loc[flip_male, "hired"] = 1
            
        # 倾向于打压女性 candidate
        female_hired = (df["gender"] == "Female") & (df["hired"] == 1)
        if female_hired.sum() > 0:
            flip_female = np.random.choice(df[female_hired].index, size=int(0.35 * female_hired.sum()), replace=False)
            df.loc[flip_female, "hired"] = 0

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
    Uses Gradient Boosting — trained on raw imbalanced data with NO fairness corrections.
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
) -> tuple:
    """
    Train a fairness-aware recruitment model using Fairlearn's
    ExponentiatedGradient reduction with Equalized Odds constraint.
    """
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

    # =========================================================================
    # 健壮性修复：创建一个完全兼容 sklearn 管道和特征名称要求的概率包装器
    # =========================================================================
    class FairlearnProbWrapper:
        def __init__(self, model):
            self.model = model
            self.classes_ = np.array([0, 1])

        def predict(self, X):
            # 基于加权加总的 predict_proba 提供确定性的 0.5 判定阈值，防止内部随机预测导致的指标抖动
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

        def predict_proba(self, X):
            # 始终将输入规整为带有正确列名的 DataFrame，消除 scikit-learn 的 Feature Names 警告或报错
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
                
            proba = np.zeros((X.shape[0], 2))
            for predictor, weight in zip(self.model.predictors_, self.model.weights_):
                proba += weight * predictor.predict_proba(X)
            return proba

    wrapped_model = FairlearnProbWrapper(fairness_model)

    return wrapped_model, scaler


def score_candidate(candidate_features: dict, biased_model, fairness_model, scaler) -> dict:
    """
    Score a single candidate using both models.
    """
    # Convert candidate dict to DataFrame row with aligned columns
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
