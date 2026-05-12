"""
helpers.py — Reusable utility functions for Diabetes ML Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Column Classification ──────────────────────────────────────────────────

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Classifies columns into: categorical, numerical, cardinal.

    Parameters
    ----------
    cat_th : int — max unique values to treat a numeric col as categorical
    car_th : int — min unique values to treat a string col as cardinal

    Returns
    -------
    cat_cols, num_cols, cat_but_car : list
    """
    cat_cols    = [c for c in dataframe.columns if dataframe[c].dtype == "O"]
    num_but_cat = [c for c in dataframe.columns if dataframe[c].nunique() < cat_th
                   and dataframe[c].dtype != "O"]
    cat_but_car = [c for c in dataframe.columns if dataframe[c].nunique() > car_th
                   and dataframe[c].dtype == "O"]

    cat_cols = [c for c in cat_cols + num_but_cat if c not in cat_but_car]
    num_cols = [c for c in dataframe.columns if dataframe[c].dtype != "O"
                and c not in num_but_cat]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables    : {dataframe.shape[1]}")
    print(f"cat_cols     : {len(cat_cols)}  → {cat_cols}")
    print(f"num_cols     : {len(num_cols)}  → {num_cols}")
    print(f"cat_but_car  : {len(cat_but_car)}")
    return cat_cols, num_cols, cat_but_car


# ── EDA ────────────────────────────────────────────────────────────────────

def check_df(dataframe, head=5):
    print("── Shape ──────────────────────────────"); print(dataframe.shape)
    print("── Types ──────────────────────────────"); print(dataframe.dtypes)
    print("── Head ───────────────────────────────"); print(dataframe.head(head))
    print("── NA Count ───────────────────────────"); print(dataframe.isnull().sum())
    print("── Quantiles ──────────────────────────")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({
        col_name : dataframe[col_name].value_counts(),
        "Ratio"  : 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.title(f"{col_name} Distribution")
        plt.tight_layout(); plt.show()


def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        dataframe[col_name].hist(bins=30, ax=ax1, color="#74b9ff", edgecolor="white")
        ax1.set_title(f"Histogram — {col_name}")
        dataframe.boxplot(column=col_name, ax=ax2)
        ax2.set_title(f"Boxplot — {col_name}")
        plt.tight_layout(); plt.show()


def target_summary_with_num(dataframe, target, col):
    print(dataframe.groupby(target).agg({col: ["mean", "median", "std"]}), end="\n\n")


def target_summary_with_cat(dataframe, target, col):
    print(pd.DataFrame({
        "Count"       : dataframe.groupby(col)[target].count(),
        "Target_Mean" : dataframe.groupby(col)[target].mean()
    }), end="\n\n")


def correlation_matrix(dataframe, cols):
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe[cols].corr(), annot=True, fmt=".2f",
                linewidths=0.5, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()


# ── Outlier ─────────────────────────────────────────────────────────────────

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    Q1, Q3 = dataframe[col_name].quantile(q1), dataframe[col_name].quantile(q3)
    iqr = Q3 - Q1
    return Q1 - 1.5 * iqr, Q3 + 1.5 * iqr


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    return bool(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0])


def replace_with_thresholds(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    dataframe[col_name] = dataframe[col_name].astype(float)   # ← bu satırı ekle
    dataframe.loc[dataframe[col_name] < low, col_name] = low
    dataframe.loc[dataframe[col_name] > up,  col_name] = up

# ── Encoding ────────────────────────────────────────────────────────────────

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)