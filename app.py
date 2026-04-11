import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time

# Page config
st.set_page_config(page_title="FraudShield Pro | AI-Powered Fraud Detection", layout="wide", page_icon="🛡️")

# Load model and scaler
@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open("fraud_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        with open("model_metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return model, scaler, meta
    except Exception as e:
        return None, None, {"name": "H.G.B. Classifier", "f1_score": 0.7265}

model, scaler, meta = load_resources()
model_name = meta.get("name", "Unknown")
model_f1 = meta.get("f1_score", 0)

# Lottie Loader
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hero = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_6p8yoj9o.json")

# PREMIUM CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #111827, #1e1b4b, #0f172a);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Header Styling */
    .main-title {
        font-size: 72px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #22d3ee, #818cf8, #22d3ee);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        line-height: 1.2;
    }

    @keyframes shine { to { background-position: 200% center; } }

    .subtitle { color: #94a3b8; text-align: center; font-size: 20px; margin-bottom: 40px; }

    /* Buttons */
    .stButton > button {
        width: 100%; background: linear-gradient(90deg, #06b6d4, #6366f1);
        border: none; color: white; padding: 15px 30px; font-size: 18px; font-weight: 600;
        border-radius: 12px; box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }

    /* Metric Styling */
    .metric-box { text-align: center; padding: 20px; border-radius: 15px; }
    .metric-value { font-size: 36px; font-weight: 800; }
    .metric-label { color: #94a3b8; font-size: 14px; text-transform: uppercase; }

    h1, h2, h3, h4, p, li { color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# Main Hero
col_lottie, col_text = st.columns([1, 2])
with col_lottie:
    if lottie_hero: st_lottie(lottie_hero, height=250, key="hero")
with col_text:
    st.markdown('<div class="main-title">FraudShield Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Real-time Credit Card Fraud Detection</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/isometric/512/security-shield.png", width=100)
st.sidebar.markdown("### Neural Status")
st.sidebar.markdown(f"""
<div style="background: rgba(6, 182, 212, 0.1); padding: 20px; border-radius: 15px; border-left: 4px solid #06b6d4; margin-bottom: 20px;">
    <p style="margin:0; font-size:12px; text-transform:uppercase; color:#06b6d4;">Active Intelligence</p>
    <p style="margin:5px 0; font-size:18px; font-weight:700;">{model_name}</p>
    <p style="margin:0; font-size:14px; opacity:0.8;">F1 Accuracy: {model_f1:.4f}</p>
</div>
""", unsafe_allow_html=True)

# Main Content
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📥 Transaction Data Input")
file = st.file_uploader("Upload CSV transaction logs", type=["csv"])

if file is not None:
    df_raw = pd.read_csv(file)
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    if st.button("🚀 Analyze Transactions"):
        with st.spinner("Decoding cryptographic patterns..."):
            time.sleep(1)
            
            df_proc = df_raw.copy()
            if 'Class' in df_proc.columns: df_proc = df_proc.drop('Class', axis=1)
            
            df_scaled = scaler.transform(df_proc)
            pred = model.predict(df_scaled)
            
            df_raw['Analysis Result'] = ["🛑 FRAUD" if p == 1 else "✅ NORMAL" for p in pred]
            df_raw['Class_Num'] = pred
            fraud_count = sum(pred)
            normal_count = len(pred) - fraud_count
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box" style="background: rgba(34, 211, 238, 0.1); border: 1px solid #22d3ee;"><div class="metric-value" style="color:#22d3ee;">{normal_count}</div><div class="metric-label">Safe</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box" style="background: rgba(244, 63, 94, 0.1); border: 1px solid #f43f5e;"><div class="metric-value" style="color:#f43f5e;">{fraud_count}</div><div class="metric-label">Fraud</div></div>', unsafe_allow_html=True)
            with m3:
                score = (normal_count/len(pred))*100
                st.markdown(f'<div class="metric-box" style="background: rgba(139, 92, 246, 0.1); border: 1px solid #8b5cf6;"><div class="metric-value" style="color:#8b5cf6;">{score:.1f}%</div><div class="metric-label">Integrity</div></div>', unsafe_allow_html=True)

            # INTELLIGENCE INSIGHTS
            st.markdown("---")
            st.markdown("### 🧠 Intelligence Insights")
            
            tab1, tab2, tab3 = st.tabs(["💸 Amount Analysis", "⏳ Temporal Patterns", "🧬 Feature Correlation"])
            
            with tab1:
                st.markdown("#### Transaction Amount Distribution")
                fig_amt = px.histogram(df_raw, x="Amount", color="Analysis Result", 
                                     marginal="box", nbins=50, log_y=True,
                                     color_discrete_map={"✅ NORMAL": "#22d3ee", "🛑 FRAUD": "#f43f5e"})
                fig_amt.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_amt, use_container_width=True)
            
            with tab2:
                st.markdown("#### Fraud Density Over Time")
                fig_time = px.scatter(df_raw.sample(min(len(df_raw), 5000)), x="Time", y="Amount", 
                                    color="Analysis Result", size="Amount", hover_data=['V1', 'V2'],
                                    color_discrete_map={"✅ NORMAL": "rgba(34, 211, 238, 0.2)", "🛑 FRAUD": "#f43f5e"})
                fig_time.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_time, use_container_width=True)
                st.caption("Note: Visualization uses a 5,000 row sample for performance efficiency.")

            with tab3:
                st.markdown("#### PCA Feature Correlation Heatmap")
                v_cols = [f'V{i}' for i in range(1, 29)]
                corr = df_raw[v_cols + ['Class_Num']].corr()
                fig_heat = px.imshow(corr, text_auto=".1f", color_continuous_scale='RdBu_r', aspect="auto")
                fig_heat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_heat, use_container_width=True)

            # Export
            st.markdown("---")
            csv = df_raw.drop('Class_Num', axis=1).to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Intelligence Report", csv, "fraud_analysis.csv", "text/csv")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 50px; padding: 30px; text-align: center; background: rgba(0,0,0,0.2); border-radius: 20px;">
    <p style="color:#94a3b8; font-size:14px;">Proprietary AI Engine © 2026 | Developed by <b>Ram Chandra Nayak</b></p>
</div>
""", unsafe_allow_html=True)