import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Stroke Risk Prediction System")
st.markdown(
    """
Welcome to the **Stroke Risk Prediction App**, an end-to-end Data Science and Machine Learning project designed to assess a patient's risk of suffering a stroke based on clinical and lifestyle parameters.

#### 🔍 Key Features & Methodology:
* **Leakage-Free Preprocessing Pipeline:** Built using a unified `ColumnTransformer` and `Pipeline` featuring median imputation, One-Hot encoding for nominal variables, and `StandardScaler` for continuous clinical features.
* **Imbalance-Aware Model Selection:** Tuned multiple classification algorithms via Stratified Cross-Validation using **PR-AUC (`average_precision`)**, optimizing the decision threshold for the **F2-Score** strictly on out-of-fold training predictions to prioritize recall without overfitting the test set.
* **Probability Calibration:** Wrapped the final **LogisticRegression** pipeline in `CalibratedClassifierCV` so that predicted probabilities reflect true clinical risk prevalence.

*Navigate to the **Prediction** page from the sidebar to evaluate a patient profile.*
"""
)
st.divider()


@st.cache_data
def load_scores():
    with open("models_score.json", "r") as f:
        return json.load(f)


data = load_scores()
df_scores = pd.DataFrame.from_dict(data, orient="index")
st.subheader("Model Comparison on Held-Out Test Set")
st.dataframe(df_scores, use_container_width=True)

st.markdown(
    """
***Model Selection Rationale***: Models were evaluated on the held-out test set using the **F2-Score**, which weighs **Recall** twice as heavily as **Precision**—a critical requirement in medical screening where missing a true stroke case (false negative) carries severe clinical consequences.

* Complex tree-based ensembles like **XGBoost** (F2: `0.3784`, Recall: `0.56`) suffered from overfitting due to the low absolute number of positive cases in the dataset, missing 44% of stroke events on unseen data.
* Linear and margin-based classifiers (**Support Vector Machine** with F2 `0.4493` and **Logistic Regression** with F2 `0.4454`) generalized significantly better.
* **Logistic Regression** was selected as the final production model: it achieves virtually the same F2-Score as SVM while providing a **higher Recall (80% vs. 78%)**, superior probability stability, and direct clinical interpretability.
"""
)