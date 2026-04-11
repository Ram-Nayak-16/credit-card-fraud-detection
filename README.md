# 🛡️ FraudShield Pro: AI-Powered Fraud Detection

![GitHub Version](https://img.shields.io/github/v/release/Ram-Nayak-16/credit-card-fraud-detection?color=blue&label=version)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://fraudshield-pro.streamlit.app/)

**FraudShield Pro** is a premium, high-intelligence dashboard designed to detect fraudulent credit card transactions in real-time. Built with a sophisticated multi-algorithm machine learning pipeline and a state-of-the-art Glassmorphism UI, it provides both powerful detection and deep analytical insights.

---

## ⚡ Key Features

- **🏆 Multi-Algorithm Intelligence**: Compares Logistic Regression, Random Forest, Decision Trees, and Hist Gradient Boosting to ensure the highest F1-Score.
- **⚖️ SMOTE Rebalancing**: Uses Synthetic Minority Over-sampling to handle extreme class imbalance (0.17% fraud).
- **✨ Glassmorphism UI**: A stunning, responsive dashboard with animated backgrounds and Lottie graphics.
- **🧠 Intelligence Insights**: Advanced analytical tabs for:
  - **Amount Analysis**: Distribution of transaction values (Log-scale).
  - **Temporal Patterns**: Scatter tracking of fraud density over time.
  - **Feature Correlation**: Deep-dive into PCA feature relationships.
- **💾 Intelligence Reports**: Export comprehensive analysis results directly to CSV.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Pip package manager

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Ram-Nayak-16/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional)**
   The best-performing model is already included, but you can retrain the pipeline using:
   ```bash
   python train_model.py
   ```

4. **Launch the Dashboard**
   ```bash
   python -m streamlit run app.py
   ```

---

## 📊 Performance Metrics

| Algorithm | F1-Score | Accuracy |
| :--- | :--- | :--- |
| **Hist Gradient Boosting** | **0.7265** | **99.89%** |
| Random Forest | 0.4914 | 99.69% |
| Logistic Regression | 0.2334 | 98.99% |
| Decision Tree | 0.1609 | 98.43% |

*Note: F1-Score is optimized for fraud detection to minimize missed cases.*

---

## 🛡️ Privacy & Security
This application processes all data locally on your machine or within your private instance. No transaction data is stored or transmitted externally beyond the Lottie animation URLs.

---

## 👨‍💻 Developed By
**Ram Chandra Nayak**  
🔗 [LinkedIn](https://www.linkedin.com/in/ram-chandra-nayak-7594a121a)  
📧 [Email](mailto:ramnayak778800@gmail.com)
