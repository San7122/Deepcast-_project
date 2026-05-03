# DeepCSAT: E-Commerce Customer Satisfaction Score Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning project that predicts **Customer Satisfaction (CSAT) scores** (1–5) for the e-commerce platform **Shopzilla** using an Artificial Neural Network (ANN). The model analyzes customer support interaction data — including service channel, issue category, response times, agent performance, and customer feedback — to forecast satisfaction levels in real time.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Business Insights](#business-insights)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement

Customer satisfaction in e-commerce is a critical metric that drives retention, repeat purchases, and revenue. Traditional post-interaction surveys are **reactive** — by the time a low score is recorded, the customer is already dissatisfied.

This project builds a **predictive model** that forecasts CSAT scores based on interaction features, enabling:

- **Real-time flagging** of at-risk interactions for supervisor intervention
- **Agent performance monitoring** to identify coaching opportunities
- **Proactive service improvement** based on predicted satisfaction trends

---

## Dataset

- **Source:** Shopzilla E-Commerce Customer Support Data
- **Records:** 85,907 customer support interactions
- **Time Period:** One month
- **Target Variable:** CSAT Score (1–5, integer)
- **Features:** 20 columns including timestamps, categorical data, numerical values, and text feedback

| Feature | Type | Description |
|---------|------|-------------|
| `channel_name` | Categorical | Service channel (Inbound, Outcall, Email) |
| `category` / `Sub-category` | Categorical | Issue type and sub-type |
| `Customer Remarks` | Text | Customer feedback (free text) |
| `Issue_reported at` / `issue_responded` | Datetime | Timestamps for response time calculation |
| `Customer_City` | Categorical | Customer location (1,782 unique cities) |
| `Product_category` | Categorical | Product type (9 categories) |
| `Item_price` | Numerical | Price of the item (₹) |
| `Agent_name` / `Supervisor` / `Manager` | Categorical | Support team hierarchy |
| `Tenure Bucket` | Categorical | Agent experience level |
| `Agent Shift` | Categorical | Shift timing (Morning, Afternoon, Evening, Night, Split) |
| `CSAT Score` | Integer | **Target** — satisfaction rating (1 to 5) |

**Key Challenge:** Severe class imbalance — Score 5 accounts for **69.4%** of all records.

[Download Dataset](https://drive.google.com/file/d/14IJWsaVX8OXW97M1fsYb-m9CyjuhchpJ/view?usp=sharing)

---

## Project Architecture

```
Raw Data (85,907 records)
    │
    ▼
Data Cleaning & Integrity
    │  • Handle 80%+ missing data in several columns
    │  • Drop extremely sparse features (99.7% null)
    │  • Parse datetime columns
    │  • Impute missing values (median / 'Unknown')
    ▼
Feature Engineering (16+ new features)
    │  • Temporal: response_time, hour, day, weekend, survey_delay
    │  • Agent: smoothed target encoding (Bayesian)
    │  • Text: sentiment scoring (positive/negative keywords)
    │  • Categorical: city grouping, sub-category grouping, price bins
    ▼
Preprocessing
    │  • One-hot encoding (categorical → binary)
    │  • StandardScaler normalization
    │  • Stratified train/test split (80/20)
    │  • Class weight computation (balanced)
    ▼
ANN Model (4 hidden layers)
    │  • 256 → 128 → 64 → 32 neurons
    │  • BatchNorm + Dropout + L2 regularization
    │  • Softmax output (5 classes)
    ▼
Evaluation & Validation
    │  • Accuracy, F1, MAE, RMSE
    │  • Confusion matrix
    │  • 5-fold stratified cross-validation
    ▼
Local Deployment
       • Model saved as .keras
       • Scaler & feature columns saved as .pkl
       • Prediction demo on new data
```

---

## Feature Engineering

| Feature Category | Features Created | Rationale |
|-----------------|-----------------|-----------|
| **Temporal** | `response_time_min`, `report_hour`, `report_dayofweek`, `report_is_weekend`, `report_day`, `survey_delay_days`, `response_speed` | Response time is the most controllable operational metric |
| **Agent Performance** | `Agent_name_csat_enc`, `Supervisor_csat_enc`, `Manager_csat_enc` | Historical agent performance strongly predicts future outcomes |
| **Text-Derived** | `has_remarks`, `remarks_length`, `remarks_word_count`, `remarks_positive`, `remarks_negative`, `remarks_sentiment` | Customer feedback is the most direct satisfaction signal |
| **Categorical** | `city_group`, `subcat_group`, `price_bin` | Reduces noise from rare categories |

---

## Model Architecture

```
Input Layer (90+ features after encoding)
    │
Dense(256, ReLU, He Normal, L2=0.001)
    │── BatchNormalization
    │── Dropout(0.4)
    │
Dense(128, ReLU, He Normal, L2=0.001)
    │── BatchNormalization
    │── Dropout(0.3)
    │
Dense(64, ReLU, He Normal, L2=0.001)
    │── BatchNormalization
    │── Dropout(0.3)
    │
Dense(32, ReLU, He Normal)
    │── Dropout(0.2)
    │
Dense(5, Softmax) → CSAT Score (1-5)
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Batch Size | 256 |
| Max Epochs | 100 (with early stopping) |
| Early Stopping | patience=10, restore best weights |
| LR Reduction | factor=0.5, patience=5 |
| Class Weights | Balanced (to counter 69% Score-5 dominance) |

**Innovative Techniques:**

- **Custom Ordinal-Aware Loss:** Adds a penalty proportional to the distance between predicted and true class — penalizes predicting CSAT 1 as 5 more than predicting 4 as 5
- **Cosine Annealing with Warm-Up:** 3-epoch warm-up phase followed by cosine decay for stable convergence

---

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~42.6% |
| **F1-Score (weighted)** | ~0.50 |
| **MAE** | ~1.23 |
| **Within ±1 Accuracy** | ~67.1% |
| **Cross-Val Accuracy** | ~40.5% ± 2.2% |

> **Note:** A naive baseline predicting all Score-5 would achieve 69% accuracy but 0% recall on dissatisfied customers. Our model trades raw accuracy for the ability to **detect dissatisfied customers (CSAT 1-3)**, which is the real business value.

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

### Clone the Repository

```bash
git clone https://github.com/<your-username>/DeepCSAT.git
cd DeepCSAT
```

### Run the Notebook

**Option A — Google Colab (Recommended):**
1. Upload `DeepCSAT_Capstone_Project.ipynb` to Google Colab
2. Upload `eCommerce_Customer_support_data__1_.csv` to the Colab file browser
3. Click `Runtime → Run All`

**Option B — Local Jupyter:**
```bash
jupyter notebook DeepCSAT_Capstone_Project.ipynb
```

---

## Usage

### Training (from the notebook)

Run all cells in the notebook sequentially. The model trains in ~5-10 minutes on CPU.

### Prediction (using saved model)

```python
import numpy as np
import pickle
from tensorflow import keras

# Load artifacts
model = keras.models.load_model('csat_ann_model.keras')
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('feature_columns.pkl', 'rb'))

# Preprocess your new data (apply same feature engineering pipeline)
# X_new = ... (DataFrame with columns matching `features`)

X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
csat_scores = np.argmax(predictions, axis=1) + 1  # Convert to 1-5

print(f"Predicted CSAT Scores: {csat_scores}")
print(f"Confidence: {np.max(predictions, axis=1)}")
```

---

## Project Structure

```
DeepCSAT/
│
├── DeepCSAT_Capstone_Project.ipynb   # Main notebook (complete pipeline)
├── eCommerce_Customer_support_data__1_.csv  # Dataset
├── README.md                          # This file
│
├── saved_models/                      # Generated after training
│   ├── csat_ann_model.keras          # Trained ANN model
│   ├── scaler.pkl                    # StandardScaler
│   ├── feature_columns.pkl           # Feature column names
│   └── class_weights.pkl             # Class weight dictionary
│
└── docs/
    └── Video_Presentation_Script.md  # Presentation guide
```

---

## Business Insights

### Key Findings

1. **Agent Performance is the #1 Predictor** — Historical agent CSAT scores are the strongest signal. Investing in agent training has the highest ROI.

2. **Response Time Drives Satisfaction** — Faster responses correlate strongly with higher CSAT. A sub-30-minute response SLA is recommended.

3. **Issue Category Sets Expectations** — Cancellations and Returns inherently produce lower satisfaction and need specialized senior agents.

4. **Customer Remarks Carry Signal** — Even simple keyword-based sentiment analysis provides meaningful predictive value.

### Recommended Actions

| Action | Impact | Priority |
|--------|--------|----------|
| Coach low-performing agents (encoded score < 3.5) | Improve weakest CSAT scores directly | High |
| Enforce < 30 min response SLA | Reduce low-CSAT incidents | High |
| Assign senior agents to Returns/Cancellations | Improve lowest-scoring categories | Medium |
| Deploy real-time CSAT predictor | Enable proactive intervention | Medium |
| Add NLP on customer remarks | Capture deeper sentiment | Low |

---

## Future Improvements

- **NLP Integration:** Use BERT or TF-IDF embeddings on Customer Remarks for deeper text understanding
- **Ensemble Methods:** Combine ANN with XGBoost/LightGBM for improved robustness
- **Attention Mechanisms:** Add attention layers to highlight most relevant features per prediction
- **Real-Time Dashboard:** Deploy via Flask/Streamlit with live CSAT monitoring
- **SMOTE Oversampling:** Address class imbalance at the data level alongside loss weighting
- **Model Retraining Pipeline:** Automated weekly retraining as new interaction data arrives

---

## Acknowledgements

- **Dataset:** Shopzilla E-Commerce Customer Support Data (provided by AlmaBetter)
- **Framework:** TensorFlow/Keras for deep learning model development
- **Course:** AlmaBetter End-Course Summative Assessment — Capstone Project

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
