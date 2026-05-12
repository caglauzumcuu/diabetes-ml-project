"""
pipeline.py — End-to-End Diabetes ML Pipeline
Steps:
    1. Data Preprocessing & Feature Engineering
    2. Base Models
    3. Hyperparameter Optimization
    4. Voting Classifier (Ensemble)
    5. Save Model
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from helpers import (
    grab_col_names, one_hot_encoder,
    replace_with_thresholds, check_outlier
)

RANDOM_STATE = 42


# ── 1. Data Preprocessing & Feature Engineering ─────────────────────────────

def diabetes_data_prep(dataframe):
    """
    Full preprocessing pipeline for PIMA Diabetes dataset.

    New Features
    ------------
    NEW_GLUCOSE_CAT  : normal / prediabetes
    NEW_AGE_CAT      : young / middleage / old
    NEW_BMI_RANGE    : underweight / healthy / overweight / obese
    NEW_BLOODPRESSURE: normal / hs1 / hs2
    NEW_GLUCOSE_BMI  : glucose × BMI interaction
    NEW_INSULIN_BMI  : insulin × BMI interaction

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    df = dataframe.copy()
    df.columns = [c.upper() for c in df.columns]

    # Replace impossible zeros with NaN
    zero_cols = ["GLUCOSE", "BLOODPRESSURE", "SKINTHICKNESS", "INSULIN", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    # Impute with group-wise median (grouped by Outcome)
    for col in zero_cols:
        df[col] = df.groupby("OUTCOME")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Binned / categorical features
    df["NEW_GLUCOSE_CAT"] = pd.cut(
        df["GLUCOSE"], bins=[-1, 139, 200],
        labels=["normal", "prediabetes"]
    )
    df["NEW_AGE_CAT"] = pd.cut(
        df["AGE"], bins=[0, 35, 55, 100],
        labels=["young", "middleage", "old"]
    )
    df["NEW_BMI_RANGE"] = pd.cut(
        df["BMI"], bins=[-1, 18.5, 24.9, 29.9, 100],
        labels=["underweight", "healthy", "overweight", "obese"]
    )
    df["NEW_BLOODPRESSURE"] = pd.cut(
        df["BLOODPRESSURE"], bins=[-1, 79, 89, 123],
        labels=["normal", "hs1", "hs2"]
    )

    # Interaction features
    df["NEW_GLUCOSE_BMI"] = df["GLUCOSE"] * df["BMI"]
    df["NEW_INSULIN_BMI"] = df["INSULIN"] * df["BMI"]

    # Encode categoricals
    cat_cols, num_cols, _ = grab_col_names(df, cat_th=5, car_th=20)
    cat_cols = [c for c in cat_cols if "OUTCOME" not in c]
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Re-detect after encoding
    cat_cols, num_cols, _ = grab_col_names(df, cat_th=5, car_th=20)

    # Outlier capping — all numeric columns
    for col in num_cols:
        replace_with_thresholds(df, col)

    # Standard scaling
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # get_dummies bool → int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    y = df["OUTCOME"]
    X = df.drop("OUTCOME", axis=1)
    return X, y


# ── 2. Base Models ───────────────────────────────────────────────────────────

def base_models(X, y, scoring="roc_auc"):
    """Trains and cross-validates all classifiers with default params."""
    print("\n" + "=" * 50)
    print("BASE MODELS")
    print("=" * 50)

    classifiers = [
        ("LR",       LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("KNN",      KNeighborsClassifier()),
        ("SVC",      SVC(probability=True, random_state=RANDOM_STATE)),
        ("CART",     DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("RF",       RandomForestClassifier(random_state=RANDOM_STATE)),
        ("AdaBoost", AdaBoostClassifier(random_state=RANDOM_STATE)),
        ("GBM",      GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ("XGBoost",  XGBClassifier(eval_metric="logloss",
                                   random_state=RANDOM_STATE, verbosity=0)),
        ("LightGBM", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
    ]

    results = {}
    for name, clf in classifiers:
        cv = cross_validate(clf, X, y, cv=5, scoring=scoring)
        score = round(cv["test_score"].mean(), 4)
        results[name] = score
        print(f"  {name:<12} {scoring}: {score}")

    return results


# ── 3. Hyperparameter Optimization ──────────────────────────────────────────

# Search spaces
knn_params = {"n_neighbors": range(2, 30)}

cart_params = {
    "max_depth"        : range(1, 15),
    "min_samples_split": range(2, 20)
}

rf_params = {
    "max_depth"        : [8, 15, None],
    "max_features"     : [5, 7, "sqrt"],
    "min_samples_split": [15, 20],
    "n_estimators"     : [200, 300]
}

xgboost_params = {
    "learning_rate"  : [0.1, 0.01],
    "max_depth"      : [5, 8],
    "n_estimators"   : [100, 200],
    "colsample_bytree": [0.5, 1.0]
}

lightgbm_params = {
    "learning_rate"  : [0.01, 0.1],
    "n_estimators"   : [300, 500],
    "colsample_bytree": [0.7, 1.0]
}

TUNED_CLASSIFIERS = [
    ("KNN",      KNeighborsClassifier(),                                   knn_params),
    ("CART",     DecisionTreeClassifier(random_state=RANDOM_STATE),        cart_params),
    ("RF",       RandomForestClassifier(random_state=RANDOM_STATE),        rf_params),
    ("XGBoost",  XGBClassifier(eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0),   xgboost_params),
    ("LightGBM", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),   lightgbm_params),
]


def hyperparameter_optimization(X, y, cv=5, scoring="roc_auc"):
    """GridSearchCV for each classifier. Returns best fitted models."""
    print("\n" + "=" * 50)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)

    best_models = {}
    for name, clf, params in TUNED_CLASSIFIERS:
        print(f"\n── {name} ──────────────────────────────")

        before = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        print(f"  {scoring} before: {round(before['test_score'].mean(), 4)}")

        gs = GridSearchCV(clf, params, cv=cv, n_jobs=-1, verbose=0).fit(X, y)
        best_clf = clf.set_params(**gs.best_params_)

        after = cross_validate(best_clf, X, y, cv=cv, scoring=scoring)
        print(f"  {scoring} after : {round(after['test_score'].mean(), 4)}")
        print(f"  Best params    : {gs.best_params_}")

        best_models[name] = best_clf

    return best_models


# ── 4. Voting Classifier ─────────────────────────────────────────────────────

def voting_classifier(best_models, X, y):
    """
    Soft voting ensemble: KNN + RF + LightGBM.
    Prints Accuracy, F1, ROC-AUC.
    """
    print("\n" + "=" * 50)
    print("VOTING CLASSIFIER (Ensemble)")
    print("=" * 50)

    voting_clf = VotingClassifier(
        estimators=[
            ("KNN",      best_models["KNN"]),
            ("RF",       best_models["RF"]),
            ("LightGBM", best_models["LightGBM"]),
        ],
        voting="soft"
    ).fit(X, y)

    cv = cross_validate(
        voting_clf, X, y, cv=5,
        scoring=["accuracy", "f1", "roc_auc"]
    )
    print(f"  Accuracy : {round(cv['test_accuracy'].mean(), 4)}")
    print(f"  F1 Score : {round(cv['test_f1'].mean(), 4)}")
    print(f"  ROC-AUC  : {round(cv['test_roc_auc'].mean(), 4)}")

    return voting_clf


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    print("Pipeline started...\n")

    base_dir = Path(__file__).parent.parent
    df = pd.read_csv(base_dir / "datasets" / "diabetes.csv")

    X, y = diabetes_data_prep(df)
    print(f"X shape: {X.shape} | y shape: {y.shape}")

    base_models(X, y)

    best_models = hyperparameter_optimization(X, y)

    voting_clf = voting_classifier(best_models, X, y)

    model_path = "../models/voting_clf.pkl"
    joblib.dump(voting_clf, model_path)
    print(f"\nModel saved → {model_path}")

    return voting_clf


if __name__ == "__main__":
    main()