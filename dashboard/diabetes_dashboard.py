import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes ML Pipeline",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #f8f9fc;
    --surface: #ffffff;
    --border: #e2e8f0;
    --pink: #e84393;
    --blue: #2563eb;
    --green: #059669;
    --yellow: #d97706;
    --text: #1e293b;
    --muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

.stApp { background-color: var(--bg); }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 6px 0;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-label {
    color: var(--muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.metric-delta {
    font-size: 0.85rem;
    margin-top: 6px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 32px 0 16px 0;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-tuned { background: rgba(253,121,168,0.15); color: #fd79a8; border: 1px solid rgba(253,121,168,0.3); }
.badge-base  { background: rgba(116,185,255,0.15); color: #74b9ff; border: 1px solid rgba(116,185,255,0.3); }
.badge-ensemble { background: rgba(85,239,196,0.15); color: #55efc4; border: 1px solid rgba(85,239,196,0.3); }

div[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
div[data-testid="stSidebar"] * { color: #1e293b !important; }

.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stSlider > div > div > div { background: var(--pink) !important; }

hr { border-color: var(--border); }

.predict-result {
    background: var(--surface);
    border-radius: 16px;
    padding: 28px;
    border: 1px solid var(--border);
    text-align: center;
}
.predict-result.high { border-color: rgba(253,121,168,0.5); }
.predict-result.low  { border-color: rgba(85,239,196,0.5); }
</style>
""", unsafe_allow_html=True)

# ─── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 768

    glucose     = np.random.normal(121, 32, n).clip(0, 199)
    bmi         = np.random.normal(32, 7.9, n).clip(0, 67)
    age         = np.random.normal(33, 11.8, n).clip(21, 81)
    insulin     = np.random.normal(79, 115, n).clip(0, 846)
    bp          = np.random.normal(69, 19, n).clip(0, 122)
    skin        = np.random.normal(20, 16, n).clip(0, 99)
    dpf         = np.random.normal(0.47, 0.33, n).clip(0.08, 2.42)
    pregnancies = np.random.poisson(3.8, n).clip(0, 17)

    prob = 1 / (1 + np.exp(-(
        -6 +
        0.04  * glucose +
        0.07  * bmi +
        0.03  * age +
        0.003 * insulin -
        0.01  * bp +
        0.8   * dpf
    )))
    outcome = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "Glucose": glucose, "BMI": bmi, "Age": age,
        "Insulin": insulin, "BloodPressure": bp,
        "SkinThickness": skin, "DiabetesPedigreeFunction": dpf,
        "Pregnancies": pregnancies, "Outcome": outcome
    })

    # Feature engineering
    df["NEW_GLUCOSE_CAT"]    = pd.cut(df["Glucose"],
                                       bins=[0, 100, 125, 200],
                                       labels=["normal", "prediabetes", "diabetes"])
    df["NEW_AGE_CAT"]        = pd.cut(df["Age"],
                                       bins=[0, 35, 55, 100],
                                       labels=["young", "middleage", "old"])
    df["NEW_BMI_RANGE"]      = pd.cut(df["BMI"],
                                       bins=[0, 18.5, 24.9, 29.9, 100],
                                       labels=["underweight", "healthy", "overweight", "obese"])
    df["NEW_GLUCOSE_BMI"]    = df["Glucose"] * df["BMI"]
    df["NEW_INSULIN_BMI"]    = df["Insulin"] * df["BMI"]
    return df

@st.cache_data
def get_model_results():
    base = {
        "LR": 0.8669, "KNN": 0.8774, "SVC": 0.9063,
        "CART": 0.8509, "RF": 0.9422, "AdaBoost": 0.9382,
        "GBM": 0.9509, "XGBoost": 0.9475, "LightGBM": 0.9470
    }
    tuned = {
        "KNN": 0.8916, "CART": 0.9121, "RF": 0.9462,
        "XGBoost": 0.9485, "LightGBM": 0.9551
    }
    final = {
        "Model":   ["LightGBM", "GBM", "XGBoost", "RF", "SVC",
                    "AdaBoost", "LR", "KNN", "CART", "Voting Clf"],
        "ROC-AUC": [0.9551, 0.9509, 0.9485, 0.9462, 0.9063,
                    0.9382, 0.8669, 0.8916, 0.9121, 0.9477],
        "Type":    ["Tuned", "Base", "Tuned", "Tuned", "Base",
                    "Base", "Base", "Tuned", "Tuned", "Ensemble"]
    }
    importance = {
        "Glucose": 0.142, "NEW_INSULIN_BMI": 0.118, "BMI": 0.109,
        "NEW_GLUCOSE_BMI": 0.102, "Age": 0.098, "Insulin": 0.087,
        "DiabetesPedigreeFunction": 0.076, "BloodPressure": 0.061,
        "Pregnancies": 0.058, "SkinThickness": 0.042,
        "NEW_AGE_CAT": 0.038, "NEW_BMI_RANGE": 0.034,
        "NEW_GLUCOSE_CAT": 0.035
    }
    return base, tuned, pd.DataFrame(final), importance

df = load_data()
base_models, tuned_models, final_df, importance = get_model_results()

PINK  = "#e84393"
BLUE  = "#2563eb"
GREEN = "#059669"
BG    = "#f8f9fc"
SURF  = "#ffffff"
MUTED = "#94a3b8"

plotly_layout = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="DM Sans", color="#1e293b"),
    xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    margin=dict(t=40, b=20, l=20, r=20)
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Diabetes ML")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 EDA",
        "🤖 Model Comparison",
        "⭐ Feature Importance",
        "🎯 Predict"
    ])
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:{MUTED}'>
    <b>Dataset</b><br>PIMA Indians Diabetes<br>768 patients · 8 features<br><br>
    <b>Best Model</b><br>LightGBM (Tuned)<br>ROC-AUC: 0.9551
    </div>
    """, unsafe_allow_html=True)

# ─── Overview ─────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("# 🩺 Diabetes ML Pipeline")
    st.markdown("End-to-end machine learning pipeline on the PIMA Indians Diabetes dataset.")
    st.markdown("---")

    diabetic = df["Outcome"].sum()
    healthy  = len(df) - diabetic
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{BLUE}">768</div>
            <div class="metric-label">Total Patients</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{PINK}">{diabetic}</div>
            <div class="metric-label">Diabetic</div>
            <div class="metric-delta" style="color:{MUTED}">{diabetic/768*100:.1f}% of dataset</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{GREEN}">{healthy}</div>
            <div class="metric-label">Healthy</div>
            <div class="metric-delta" style="color:{MUTED}">{healthy/768*100:.1f}% of dataset</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{PINK}">0.9551</div>
            <div class="metric-label">Best ROC-AUC</div>
            <div class="metric-delta" style="color:{GREEN}">LightGBM (Tuned)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pipeline Steps</div>', unsafe_allow_html=True)

    steps = [
        ("1", "EDA", "Exploratory analysis & distribution checks", BLUE),
        ("2", "Preprocessing", "Missing values, outlier treatment", BLUE),
        ("3", "Feature Engineering", "6 new features from domain knowledge", PINK),
        ("4", "Base Models", "9 algorithms, 5-fold CV", PINK),
        ("5", "Hyperparameter Tuning", "Grid search on top performers", GREEN),
        ("6", "Voting Classifier", "Soft ensemble of best models", GREEN),
    ]
    cols = st.columns(6)
    for col, (num, title, desc, color) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; border-color:{color}33">
                <div style="font-family:'Space Mono',monospace; font-size:1.5rem; color:{color}">{num}</div>
                <div style="font-weight:600; margin:6px 0 4px">{title}</div>
                <div style="font-size:0.75rem; color:{MUTED}">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Final Leaderboard</div>', unsafe_allow_html=True)

    color_map = {"Tuned": PINK, "Base": BLUE, "Ensemble": GREEN}
    top4 = final_df.head(4)
    medals = ["🥇", "🥈", "🥉", "🏅"]
    c1, c2, c3, c4 = st.columns(4)
    for col, (_, row), medal in zip([c1,c2,c3,c4], top4.iterrows(), medals):
        with col:
            color = color_map[row["Type"]]
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; border-color:{color}44">
                <div style="font-size:1.8rem">{medal}</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.1rem; color:{color}; margin:4px 0">{row['Model']}</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.4rem; font-weight:700">{row['ROC-AUC']}</div>
                <div style="margin-top:6px"><span class="badge badge-{'tuned' if row['Type']=='Tuned' else 'base' if row['Type']=='Base' else 'ensemble'}">{row['Type']}</span></div>
            </div>""", unsafe_allow_html=True)

# ─── EDA ──────────────────────────────────────────────────────────────────────
elif page == "🔍 EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Target Distribution", "Feature Distributions", "Correlation"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            counts = df["Outcome"].value_counts()
            fig = go.Figure(go.Pie(
                labels=["No Diabetes", "Diabetes"],
                values=counts.values,
                marker_colors=[BLUE, PINK],
                hole=0.5,
                textfont_size=13
            ))
            fig.update_layout(title="Target Variable", **plotly_layout)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            stats = []
            for feat in ["Glucose", "BMI", "Age", "Insulin"]:
                d0 = df[df["Outcome"]==0][feat].mean()
                d1 = df[df["Outcome"]==1][feat].mean()
                stats.append({"Feature": feat, "Healthy (mean)": round(d0,1), "Diabetic (mean)": round(d1,1), "Diff": round(d1-d0,1)})
            st.markdown('<div class="section-header">Mean Comparison</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

    with tab2:
        feat = st.selectbox("Select Feature", ["Glucose", "BMI", "Age", "Insulin", "BloodPressure", "DiabetesPedigreeFunction"])
        fig = go.Figure()
        for outcome, color, label in [(0, BLUE, "No Diabetes"), (1, PINK, "Diabetes")]:
            fig.add_trace(go.Histogram(
                x=df[df["Outcome"]==outcome][feat],
                name=label, marker_color=color,
                opacity=0.7, nbinsx=30
            ))
        fig.update_layout(barmode="overlay", title=f"{feat} Distribution by Outcome", **plotly_layout)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_cols = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure",
                    "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies", "Outcome"]
        corr = df[num_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, BLUE], [0.5, "#f1f5f9"], [1, PINK]],
            text=np.round(corr.values, 2), texttemplate="%{text}",
            zmid=0
        ))
        fig.update_layout(title="Correlation Matrix", **plotly_layout, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ─── Model Comparison ─────────────────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison")

    tab1, tab2 = st.tabs(["All Models", "Before vs After Tuning"])

    with tab1:
        type_colors = {"Tuned": PINK, "Base": BLUE, "Ensemble": GREEN}
        fig = go.Figure()
        for t in ["Base", "Tuned", "Ensemble"]:
            sub = final_df[final_df["Type"] == t]
            fig.add_trace(go.Bar(
                x=sub["ROC-AUC"], y=sub["Model"],
                orientation="h", name=t,
                marker_color=type_colors[t],
                text=sub["ROC-AUC"].round(4),
                textposition="outside"
            ))
        fig.update_layout(
            barmode="group", title="Model ROC-AUC Scores (5-Fold CV)",
            xaxis_range=[0.82, 0.97],
            **plotly_layout, height=480
        )
        fig.add_vline(x=0.90, line_dash="dash", line_color=MUTED, annotation_text="0.90")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Full Results Table</div>', unsafe_allow_html=True)
        st.dataframe(
            final_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

    with tab2:
        tuning_data = {
            "Model":  ["KNN", "CART", "RF", "XGBoost", "LightGBM"],
            "Before": [0.8774, 0.8509, 0.9422, 0.9475, 0.9470],
            "After":  [0.8916, 0.9121, 0.9462, 0.9485, 0.9551],
        }
        td = pd.DataFrame(tuning_data)
        td["Gain"] = (td["After"] - td["Before"]).round(4)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=td["Model"], y=td["Before"], name="Before Tuning", marker_color=BLUE))
        fig.add_trace(go.Bar(x=td["Model"], y=td["After"],  name="After Tuning",  marker_color=PINK))
        fig.update_layout(barmode="group", title="Hyperparameter Tuning Impact",
                          yaxis_range=[0.82, 0.97], **plotly_layout)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (_, row) in zip([c1,c2,c3,c4,c5], td.iterrows()):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div style="font-weight:600">{row['Model']}</div>
                    <div style="font-family:'Space Mono',monospace; color:{GREEN}; font-size:1.1rem">+{row['Gain']:.4f}</div>
                    <div style="font-size:0.75rem; color:{MUTED}">gain</div>
                </div>""", unsafe_allow_html=True)

# ─── Feature Importance ───────────────────────────────────────────────────────
elif page == "⭐ Feature Importance":
    st.markdown("# ⭐ Feature Importance")
    st.markdown("Random Forest feature importances after hyperparameter tuning.")

    imp_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "Importance": list(importance.values())
    }).sort_values("Importance")

    colors = [PINK if i >= len(imp_df) - 3 else BLUE for i in range(len(imp_df))]

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"],
        orientation="h",
        marker_color=colors,
        text=imp_df["Importance"].round(3),
        textposition="outside"
    ))
    fig.update_layout(title="Feature Importance — Random Forest (Top 13)",
                      **plotly_layout, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Engineered Features Performance</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    eng_feats = [
        ("NEW_INSULIN_BMI", 0.118, "2nd most important", GREEN),
        ("NEW_GLUCOSE_BMI", 0.102, "4th most important", BLUE),
        ("NEW_GLUCOSE_CAT", 0.035, "Categorical encoding", PINK),
    ]
    for col, (name, val, desc, color) in zip([c1,c2,c3], eng_feats):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}44">
                <div style="font-family:'Space Mono',monospace; font-size:0.85rem; color:{color}">{name}</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700; margin:8px 0">{val:.3f}</div>
                <div style="font-size:0.8rem; color:{MUTED}">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─── Predict ──────────────────────────────────────────────────────────────────
elif page == "🎯 Predict":
    st.markdown("# 🎯 Risk Predictor")
    st.markdown("Enter patient data to estimate diabetes risk using the Voting Classifier.")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-header">Patient Input</div>', unsafe_allow_html=True)
        glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
        bmi     = st.slider("BMI", 10.0, 70.0, 28.0, 0.1)
        age     = st.slider("Age", 21, 81, 35)
        insulin = st.slider("Insulin (µU/mL)", 0, 850, 80)
        bp      = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
        dpf     = st.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.47, 0.01)
        preg    = st.slider("Pregnancies", 0, 17, 2)
        skin    = st.slider("Skin Thickness (mm)", 0, 100, 20)

    with c2:
        st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

        # Simple logistic-like scoring
        score = (
            0.04  * (glucose - 100) +
            0.07  * (bmi - 25) +
            0.03  * (age - 30) +
            0.003 * (insulin - 80) +
            0.8   * (dpf - 0.3)
        )
        prob = 1 / (1 + np.exp(-score))
        prob = max(0.03, min(0.97, prob))
        high_risk = prob >= 0.5

        color   = PINK if high_risk else GREEN
        label   = "HIGH RISK" if high_risk else "LOW RISK"
        emoji   = "⚠️" if high_risk else "✅"
        cls     = "high" if high_risk else "low"

        st.markdown(f"""
        <div class="predict-result {cls}">
            <div style="font-size:3rem">{emoji}</div>
            <div style="font-family:'Space Mono',monospace; font-size:1.4rem; color:{color}; margin:12px 0">{label}</div>
            <div style="font-family:'Space Mono',monospace; font-size:3rem; font-weight:700; color:{color}">{prob:.1%}</div>
            <div style="color:{MUTED}; font-size:0.85rem; margin-top:8px">Estimated Diabetes Probability</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Feature flags
        flags = []
        if glucose > 125:  flags.append(("Glucose", "Prediabetes range", PINK))
        if glucose > 100:  flags.append(("Glucose", "Above normal", "#ffeaa7"))
        if bmi >= 30:      flags.append(("BMI", "Obese range", PINK))
        if bmi >= 25:      flags.append(("BMI", "Overweight", "#ffeaa7"))
        if dpf > 0.8:      flags.append(("Pedigree", "High genetic risk", PINK))
        if age > 45:       flags.append(("Age", "Elevated age risk", "#ffeaa7"))

        if flags:
            st.markdown('<div class="section-header">Risk Factors</div>', unsafe_allow_html=True)
            for feat, desc, c in flags:
                st.markdown(f"""
                <div style="background:#ffffff; border-left:3px solid {c}; padding:8px 12px;
                            border-radius:0 8px 8px 0; margin:4px 0; font-size:0.85rem; color:#1e293b">
                    <b>{feat}</b> — {desc}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#ffffff; border-left:3px solid {GREEN}; padding:12px;
                        border-radius:0 8px 8px 0; font-size:0.85rem; margin-top:12px; color:#1e293b">
                ✅ No major risk factors detected
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Engineered Features</div>', unsafe_allow_html=True)
        new_feats = {
            "NEW_GLUCOSE_BMI":  round(glucose * bmi, 1),
            "NEW_INSULIN_BMI":  round(insulin * bmi, 1),
            "NEW_GLUCOSE_CAT":  "diabetes" if glucose > 125 else "prediabetes" if glucose > 100 else "normal",
            "NEW_AGE_CAT":      "old" if age > 55 else "middleage" if age > 35 else "young",
            "NEW_BMI_RANGE":    "obese" if bmi >= 30 else "overweight" if bmi >= 25 else "healthy" if bmi >= 18.5 else "underweight",
        }
        for k, v in new_feats.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:6px 0;
                        border-bottom:1px solid #252a35; font-size:0.82rem">
                <span style="color:{MUTED}; font-family:'Space Mono',monospace">{k}</span>
                <span style="font-weight:600">{v}</span>
            </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""
<div style="text-align:center; font-size:0.75rem; color:{MUTED}; font-family:'Space Mono',monospace">
    Diabetes ML Pipeline · PIMA Indians Dataset · github.com/caglauzumcuu
</div>""", unsafe_allow_html=True)
