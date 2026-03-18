import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load model
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# PREMIUM CSS
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

.main {
    background-color: rgba(0,0,0,0);
}

.title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #00DBDE, #FC00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
    margin-top: 20px;
}

.footer {
    margin-top: 50px;
    padding: 25px;
    text-align: center;
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
}

a {
    color: #00FFFF;
    text-decoration: none;
    font-weight: bold;
}

a:hover {
    color: #FF00FF;
}
</style>
""", unsafe_allow_html=True)

# Title (no emoji)
st.markdown('<div class="title">Credit Card Fraud Detection</div>', unsafe_allow_html=True)

# About Section (no emoji)
st.markdown("""
<div class="card">
<h2>About This Project</h2>
<p>
This system uses Machine Learning to detect fraudulent credit card transactions.
Upload your dataset and the model analyzes patterns to classify transactions as Fraud or Normal.
</p>
<ul>
<li>Fast automated detection</li>
<li>Real-world scalable solution</li>
<li>Accurate ML predictions</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")
st.sidebar.info("Upload your dataset")

# Upload file
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Run Prediction"):

        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)

        df['Prediction'] = pred

        fraud = sum(pred)
        normal = len(pred) - fraud

        col1, col2 = st.columns(2)
        col1.metric("Normal Transactions", normal)
        col2.metric("Fraud Transactions", fraud)

        # Graph
        chart_df = pd.DataFrame({
            "Type": ["Normal", "Fraud"],
            "Count": [normal, fraud]
        })

        fig = px.bar(chart_df, x="Type", y="Count", text="Count", color="Type")
        st.plotly_chart(fig, use_container_width=True)

        # Results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Results")
        st.dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "results.csv", "text/csv")

# Footer (EMOJIS ONLY HERE)
st.markdown("""
<div class="footer">
<h3>Contact Me</h3>

<p><b>Ram Chandra Nayak</b></p>

<p>
📞 9369665818
</p>

<p>
📧 <a href="mailto:ramnayak778800@gmail.com">ramnayak778800@gmail.com</a>
</p>

<p>
🔗 <a href="https://www.linkedin.com/in/ram-chandra-nayak-7594a121a" target="_blank">
LinkedIn Profile
</a>
</p>

</div>
""", unsafe_allow_html=True)