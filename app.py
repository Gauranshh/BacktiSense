import os
import streamlit as st
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="BactiSense", page_icon="🦠", layout="wide")

# ── Load model data ────────────────────────────────────────────
@st.cache_resource
def load_data():
    with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
        model, feature_cols = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "predictions.json")) as f:
        predictions = json.load(f)
    with open(os.path.join(BASE_DIR, "model", "feature_importance.json")) as f:
        feat_imp = json.load(f)
    return model, feature_cols, predictions, feat_imp

model, feature_cols, predictions, feat_imp = load_data()
antibiotics = ['IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN']
with open(os.path.join(BASE_DIR, "model", "accuracy.json")) as f:
    accuracy_data = json.load(f)

st.subheader("📊 Model Performance")

for ab, acc in accuracy_data.items():
    st.write(f"{ab}: {acc}% accuracy")


# ── Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0f1a; }
    .stSelectbox label { color: #00e5a0 !important; font-family: monospace; }
    .metric-card { 
        background: #0d1a14; border: 1px solid #00e5a020; 
        border-radius: 10px; padding: 16px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🦠 BactiSense — AI Antibiotic Intelligence")
    st.caption("Track B · CodeCure Biohackathon · Antimicrobial Resistance Prediction")
with col2:
    st.markdown("**Model:** Random Forest (200 trees)")
    st.markdown("**Features:** Location-encoded phenotypic data")

st.divider()

# ── Input ──────────────────────────────────────────────────────
st.subheader("📍 Select Sampling Location")
location = st.selectbox("", list(predictions.keys()))

if st.button("🔬 Predict Resistance Profile", use_container_width=True):
    input_dict = {col: 0 for col in feature_cols}

    for col in feature_cols:
        if location in col:
            input_dict[col] = 1

    sample = pd.DataFrame([input_dict])

    pred = model.predict(sample)[0]
    probs = model.predict_proba(sample)

    profile = {}
    for i, ab in enumerate(antibiotics):
        profile[ab] = [int(pred[i]), round(probs[i][0][0] * 100, 2)]
    
    # ── Results grid ───────────────────────────────────────────
    st.subheader("📊 Resistance Profile")
    cols = st.columns(len(antibiotics))
    best_ab, best_prob = None, 0
    
    for i, ab in enumerate(antibiotics):
        pred, prob = profile[ab]
        is_susc = pred == 0
        with cols[i]:
            status = "✅ Susceptible" if is_susc else "❌ Resistant"
            color = "#00e5a0" if is_susc else "#ff4d6d"
            st.markdown(f"""
            <div style='background:#0d1a14;border:1px solid {color}33;
                        border-radius:10px;padding:16px;text-align:center'>
                <div style='font-family:monospace;font-size:11px;color:#5a7a6e'>{ab}</div>
                <div style='color:{color};font-weight:700;margin:8px 0'>{status}</div>
                <div style='font-family:monospace;font-size:13px;color:#8a9a8e'>{prob}%</div>
                <div style='height:4px;background:#1a2a1e;border-radius:2px;margin-top:8px'>
                    <div style='width:{prob}%;height:100%;background:{color};border-radius:2px'></div>
                </div>
            </div>""", unsafe_allow_html=True)
        if is_susc and prob > best_prob:
            best_prob, best_ab = prob, ab

    st.divider()

    # ── Recommendation ─────────────────────────────────────────
    resistant_count = sum([1 for ab in antibiotics if profile[ab][0] == 1])

    if resistant_count >= 3:
        st.error("🚨 Multi-drug resistance pattern detected")
    elif resistant_count == 2:
        st.warning("⚠️ Moderate resistance detected")
    else:
        st.success("✅ Low resistance profile")
    if best_ab:
        st.success(f"⭐ **Recommended Antibiotic: {best_ab}** — {best_prob}% susceptibility confidence")
        st.info("This recommendation is based on predicted susceptibility across all tested antibiotics using an ensemble Random Forest model trained on bacterial isolate data from Osun State, Nigeria.")
    else:
        st.error("⚠️ All antibiotics show resistance. Escalation protocol recommended.")

    st.divider()

    # ── Charts side by side ────────────────────────────────────
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Feature Importance")
        names = [f["name"] for f in feat_imp]
        values = [f["value"] for f in feat_imp]
        colors = ["#00e5a0" if location in n else "#1a4a3a" for n in names]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.6)
        ax.set_xlabel("Importance (%)", color='#8aaa9a')
        ax.tick_params(colors='#8aaa9a', labelsize=9)
        ax.spines[:].set_color('#1a3a2a')
        ax.set_title("Top Predictive Features", color='#00e5a0', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("🕸️ Resistance Network")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')

        # Draw network: center = location, spokes = antibiotics
        n = len(antibiotics)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx, cy = 0.5, 0.5
        r = 0.35

        for i, (ab, angle) in enumerate(zip(antibiotics, angles)):
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            pred, prob = profile[ab]
            color = "#00e5a0" if pred == 0 else "#ff4d6d"
            lw = 2 + prob / 40
            ax2.plot([cx, x], [cy, y], color=color, linewidth=lw, alpha=0.7)
            ax2.scatter([x], [y], color=color, s=120, zorder=5)
            ax2.text(x, y + 0.06 * np.sign(np.sin(angle) + 0.01),
                     ab[:3], color=color, fontsize=8,
                     ha='center', fontfamily='monospace')

        ax2.scatter([cx], [cy], color='#00b4d8', s=200, zorder=6)
        ax2.text(cx, cy - 0.08, location, color='#00b4d8',
                 fontsize=9, ha='center', fontfamily='monospace', fontweight='bold')
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        green = mpatches.Patch(color='#00e5a0', label='Susceptible')
        red = mpatches.Patch(color='#ff4d6d', label='Resistant')
        ax2.legend(handles=[green, red], loc='lower right',
                   facecolor='#0d1117', labelcolor='white', fontsize=8)
        ax2.set_title("Resistance Pattern Network", color='#00e5a0', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.markdown("## 🧬 About BactiSense")
st.sidebar.markdown("""
AI-powered antimicrobial resistance Clinical Decision Support System built for the **CodeCure Biohackathon — Track B**.

**Model:** Random Forest Classifier  
**Dataset:** Antimicrobial Resistance Dataset (Mendeley)  
**Locations:** IFE-T, IFE-S, OSU-T, OSU-S, OSU-C  
**Antibiotics:** Imipenem, Ceftazidime, Gentamicin, Augmentin, Ciprofloxacin
""")
st.sidebar.info("Designed for integration with hospital lab systems and genomic datasets.")
st.sidebar.divider()
st.sidebar.markdown("**Evaluation Criteria Met:**")
st.sidebar.markdown("✅ Resistance prediction model")
st.sidebar.markdown("✅ Feature importance analysis")
st.sidebar.markdown("✅ Resistance network visualization")
st.sidebar.markdown("✅ Decision-support tool")
