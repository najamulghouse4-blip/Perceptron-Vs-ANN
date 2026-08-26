import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Perceptron vs ANN | Iris Showdown",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — dark, glassy, neon-gradient theme
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #1a0b2e 0%, #0d0221 45%, #050014 100%);
    color: #f2f0ff;
}

/* Animated hero title */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #ff6ec4, #7873f5, #4adede, #ff6ec4);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
    margin-bottom: 0;
    letter-spacing: 1px;
}
@keyframes shine {
    to { background-position: 300% center; }
}
.hero-sub {
    text-align: center;
    color: #b9b3d9;
    font-size: 1.05rem;
    margin-top: 0.3rem;
    margin-bottom: 1.6rem;
    font-weight: 300;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(120, 115, 245, 0.35);
}

/* Metric-style number cards */
.metric-box {
    text-align: center;
    padding: 1.4rem 0.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(120,115,245,0.18), rgba(255,110,196,0.12));
    border: 1px solid rgba(255,255,255,0.15);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4adede, #7873f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: #cfc9f0;
    font-size: 0.95rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #150a2b 0%, #0a0416 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.stButton>button {
    background: linear-gradient(90deg, #7873f5, #ff6ec4);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.4rem;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    width: 100%;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(120,115,245,0.6);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7873f5, transparent);
    margin: 1.6rem 0;
}

.section-header {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f2f0ff;
    border-left: 4px solid #ff6ec4;
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem 0;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">🌸 PERCEPTRON vs ANN 🌸</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A head-to-head neural showdown on the classic Iris dataset — '
    'tune it, train it, watch it battle.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# DATA LOADING (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def get_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["Species"] = iris.target
    df["Species"] = df["Species"].map(
        {0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]}
    )
    df["Id"] = df.index + 1
    return df, iris.target_names


df, target_names = get_data()

# ----------------------------------------------------------------------------
# SIDEBAR — CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("Tune the experiment, then hit **Launch**.")
    st.markdown("---")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    st.markdown("#### 🧠 ANN Architecture")
    layer1 = st.slider("Layer 1 neurons", 4, 64, 10)
    layer2 = st.slider("Layer 2 neurons", 4, 64, 8)
    dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.05)
    epochs = st.slider("Epochs", 10, 200, 50, 10)
    batch_size = st.select_slider("Batch size", options=[1, 2, 5, 10, 16, 32], value=5)

    st.markdown("---")
    run = st.button("🚀 Launch Showdown")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_overview, tab_explore, tab_battle = st.tabs(
    ["📊 Overview", "🔍 Explore Data", "⚔️ Model Battle"]
)

# ---- OVERVIEW TAB ----
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Samples", "Features", "Classes", "Test Split"],
        [len(df), df.shape[1] - 2, df["Species"].nunique(), f"{int(test_size*100)}%"],
    ):
        col.markdown(
            f'<div class="metric-box"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.style.background_gradient(cmap="cool", subset=df.columns[:4]), height=280)

    st.markdown('<div class="section-header">Species Distribution</div>', unsafe_allow_html=True)
    fig = px.pie(
        df, names="Species", hole=0.55,
        color_discrete_sequence=["#7873f5", "#ff6ec4", "#4adede"],
    )
    fig.update_traces(textfont_size=14, textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f2f0ff", showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- EXPLORE TAB ----
with tab_explore:
    st.markdown('<div class="section-header">Feature Relationships</div>', unsafe_allow_html=True)
    colx, coly = st.columns(2)
    with colx:
        x_feat = st.selectbox("X axis", df.columns[:4], index=0)
    with coly:
        y_feat = st.selectbox("Y axis", df.columns[:4], index=2)

    fig2 = px.scatter(
        df, x=x_feat, y=y_feat, color="Species", size=df.columns[3],
        color_discrete_sequence=["#7873f5", "#ff6ec4", "#4adede"],
        opacity=0.85,
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,10,40,0.4)",
        font_color="#f2f0ff",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df[df.columns[:4]].corr()
    fig3 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="Purples",
        aspect="auto",
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f2f0ff",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">3D Feature Space</div>', unsafe_allow_html=True)
    fig4 = px.scatter_3d(
        df, x=df.columns[0], y=df.columns[1], z=df.columns[2],
        color="Species", color_discrete_sequence=["#7873f5", "#ff6ec4", "#4adede"],
    )
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#f2f0ff",
        scene=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---- BATTLE TAB ----
with tab_battle:
    if not run:
        st.info("👈 Configure your experiment in the sidebar, then click **Launch Showdown** to train both models.")
    else:
        X = df.drop(columns=["Species", "Id"])
        Y = df["Species"]

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(Y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size,
            stratify=y_encoded, random_state=int(random_state),
        )

        progress = st.progress(0, text="Training Perceptron...")

        per = Perceptron(random_state=int(random_state), max_iter=1000)
        per.fit(X_train, y_train)
        y_pred_per = per.predict(X_test)
        per_acc = accuracy_score(y_test, y_pred_per)
        progress.progress(35, text="Perceptron trained. Building ANN...")

        y_train_enc = to_categorical(y_train)
        y_test_enc = to_categorical(y_test)

        ann_model = Sequential([
            Dense(layer1, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(dropout_rate),
            Dense(layer2, activation="relu"),
            Dropout(dropout_rate),
            Dense(y_train_enc.shape[1], activation="softmax"),
        ])
        ann_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        progress.progress(55, text=f"Training ANN for {epochs} epochs...")
        history = ann_model.fit(
            X_train, y_train_enc, epochs=epochs, batch_size=batch_size, verbose=0
        )
        progress.progress(90, text="Evaluating...")

        loss, ann_acc = ann_model.evaluate(X_test, y_test_enc, verbose=0)
        ann_preds = ann_model.predict(X_test, verbose=0)
        ann_pred_classes = np.argmax(ann_preds, axis=1)

        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()
        st.balloons()

        # ---- Scoreboard ----
        st.markdown('<div class="section-header">🏆 Scoreboard</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        winner = "Perceptron" if per_acc > ann_acc else ("ANN" if ann_acc > per_acc else "Tie")
        with c1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{per_acc:.2%}</div>'
                f'<div class="metric-label">Perceptron Accuracy</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{ann_acc:.2%}</div>'
                f'<div class="metric-label">ANN Accuracy</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<h3 style="text-align:center; margin-top:1rem;">🎉 Winner: '
            f'<span style="background:linear-gradient(90deg,#ff6ec4,#4adede);'
            f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{winner}</span></h3>',
            unsafe_allow_html=True,
        )

        # ---- Accuracy bar chart ----
        acc_df = pd.DataFrame({"Model": ["Perceptron", "ANN"], "Accuracy": [per_acc, ann_acc]})
        fig5 = px.bar(
            acc_df, x="Model", y="Accuracy", color="Model", text_auto=".2%",
            color_discrete_sequence=["#7873f5", "#ff6ec4"], range_y=[0, 1.05],
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,10,40,0.4)",
            font_color="#f2f0ff", showlegend=False,
        )
        st.plotly_chart(fig5, use_container_width=True)

        # ---- Training curve ----
        st.markdown('<div class="section-header">📈 ANN Training Curve</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(history.history)
        fig6 = make_subplots(specs=[[{"secondary_y": True}]])
        fig6.add_trace(go.Scatter(y=hist_df["loss"], name="Loss", line=dict(color="#ff6ec4")), secondary_y=False)
        fig6.add_trace(go.Scatter(y=hist_df["accuracy"], name="Accuracy", line=dict(color="#4adede")), secondary_y=True)
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,10,40,0.4)",
            font_color="#f2f0ff", legend=dict(orientation="h", y=1.1),
        )
        fig6.update_yaxes(title_text="Loss", secondary_y=False)
        fig6.update_yaxes(title_text="Accuracy", secondary_y=True)
        st.plotly_chart(fig6, use_container_width=True)

        # ---- Confusion matrices ----
        st.markdown('<div class="section-header">🧩 Confusion Matrices</div>', unsafe_allow_html=True)
        cm1, cm2 = st.columns(2)
        cm_per = confusion_matrix(y_test, y_pred_per)
        cm_ann = confusion_matrix(y_test, ann_pred_classes)

        with cm1:
            fig7 = px.imshow(
                cm_per, text_auto=True, x=target_names, y=target_names,
                color_continuous_scale="Purples", title="Perceptron",
            )
            fig7.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f2f0ff")
            st.plotly_chart(fig7, use_container_width=True)
        with cm2:
            fig8 = px.imshow(
                cm_ann, text_auto=True, x=target_names, y=target_names,
                color_continuous_scale="Magenta", title="ANN",
            )
            fig8.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f2f0ff")
            st.plotly_chart(fig8, use_container_width=True)

        # ---- Classification reports ----
        st.markdown('<div class="section-header">📋 Classification Reports</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Perceptron**")
            report_per = classification_report(
                y_test, y_pred_per, target_names=target_names, output_dict=True
            )
            st.dataframe(pd.DataFrame(report_per).transpose().style.background_gradient(cmap="cool"))
            st.markdown("</div>", unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**ANN**")
            report_ann = classification_report(
                y_test, ann_pred_classes, target_names=target_names, output_dict=True
            )
            st.dataframe(pd.DataFrame(report_ann).transpose().style.background_gradient(cmap="cool"))
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#8a84b8; font-size:0.85rem;">'
    "Built with Streamlit • scikit-learn • TensorFlow/Keras • Plotly"
    "</div>",
    unsafe_allow_html=True,
)
