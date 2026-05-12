# 🩺 Diabetes End-to-End Machine Learning Pipeline

A complete machine learning pipeline for diabetes prediction using the PIMA Indians Diabetes Dataset.

---

## 📁 Project Structure
```
diabetes-ml-project/
├── notebooks/
│   └── diabetes_pipeline.ipynb   # Main notebook (EDA + ML pipeline)
├── src/
│   ├── helpers.py                # Utility functions
│   └── pipeline.py               # ML pipeline functions
├── datasets/
│   └── diabetes.csv              # Raw dataset (not tracked by git)
├── models/
│   └── voting_clf.pkl            # Trained model (not tracked by git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Source:** [PIMA Indians Diabetes Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Rows:** 768 patients
- **Features:** 8 original + 10 engineered = 18 total
- **Target:** `Outcome` (1 = Diabetic, 0 = Healthy)
- **Class balance:** 65.1% healthy / 34.9% diabetic

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-ml-project.git
cd diabetes-ml-project
```

### 2. Create virtual environment
```bash
python -m venv diabetes_venv
source diabetes_venv/bin/activate      # Mac/Linux
diabetes_venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Download `diabetes.csv` from Kaggle and place it in the `datasets/` folder.

### 5. Run the pipeline
```bash
cd src
python pipeline.py
```

### 6. Open the notebook
```bash
cd ..
jupyter notebook notebooks/diabetes_pipeline.ipynb
```

---

## 🔬 Pipeline Steps

### 1. Exploratory Data Analysis
- Target variable distribution
- Feature distributions by outcome
- Correlation matrix
- Boxplot analysis

### 2. Data Preprocessing & Feature Engineering
| Original | New Feature | Description |
|----------|-------------|-------------|
| Glucose | NEW_GLUCOSE_CAT | normal / prediabetes |
| Age | NEW_AGE_CAT | young / middleage / old |
| BMI | NEW_BMI_RANGE | underweight / healthy / overweight / obese |
| BloodPressure | NEW_BLOODPRESSURE | normal / hs1 / hs2 |
| Glucose × BMI | NEW_GLUCOSE_BMI | interaction feature |
| Insulin × BMI | NEW_INSULIN_BMI | interaction feature |

### 3. Base Model Comparison (5-Fold CV)
| Model | ROC-AUC |
|-------|---------|
| GBM | 0.9509 |
| XGBoost | 0.9475 |
| LightGBM | 0.9470 |
| RF | 0.9422 |
| AdaBoost | 0.9382 |

### 4. Hyperparameter Optimization
| Model | Before | After | Gain |
|-------|--------|-------|------|
| LightGBM | 0.9470 | 0.9551 | +0.0081 |
| CART | 0.8509 | 0.9121 | +0.0612 |
| RF | 0.9422 | 0.9462 | +0.0040 |

### 5. Voting Classifier (Ensemble)
Soft voting ensemble of KNN + RF + LightGBM.

| Metric | Score |
|--------|-------|
| Accuracy | 0.8894 |
| F1 Score | 0.8396 |
| ROC-AUC | 0.9477 |

---

## 🏆 Results

| | Model | ROC-AUC |
|--|-------|---------|
| 🥇 | LightGBM (Tuned) | 0.9551 |
| 🥈 | GBM (Base) | 0.9509 |
| 🥉 | XGBoost (Tuned) | 0.9485 |
| 🏅 | Voting Classifier | 0.9477 |

---

## 💡 Key Takeaways

- **Glucose** is the strongest predictor of diabetes (correlation: 0.47)
- **Feature engineering paid off** — NEW_INSULIN_BMI ranked 2nd in feature importance
- **Tree-based models** dominate this dataset
- **Voting Classifier** provides stable and reliable predictions across folds

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-lightblue)

---

## 👤 Author

**Çağla Üzümcü**  
[GitHub](https://github.com/caglauzumcuu)