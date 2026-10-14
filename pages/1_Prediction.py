import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stroke Prediction", page_icon="🩺")

st.title("🩺 Stroke Risk Prediction")
st.markdown(
    "Complete the clinical form below to estimate the patient's calibrated probability of experiencing a stroke."
)


@st.cache_resource
def load_model_bundle():
    return joblib.load("stroke_model_bundle.pkl")


bundle = load_model_bundle()
model = bundle["model"]
threshold = bundle["threshold"]

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (years)", min_value=1, max_value=115, value=50, step=1)
        height = st.number_input(
            "Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5
        )
        weight = st.number_input(
            "Weight (kg)", min_value=5.0, max_value=300.0, value=70.0, step=0.5
        )
        ever_married = st.radio("Have you ever been married?", ["Yes", "No"])

    with col2:
        hp = st.radio("History of hypertension?", ["No", "Yes"])
        hd = st.radio("History of heart disease?", ["No", "Yes"])
        work_type = st.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input(
            "Average glucose level (mg/dL)",
            min_value=40.0,
            max_value=400.0,
            value=90.0,
            step=1.0,
        )
        smoking_status = st.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )

    submitted = st.form_submit_button("Predict Stroke Risk")

if submitted:
    # Controllo di coerenza logica sugli input
    if age >= 18 and work_type == "children":
        st.warning(
            "⚠️ Logical warning: 'children' work type selected for an adult patient (Age >= 18). Please verify the work type."
        )
    if age < 16 and ever_married == "Yes":
        st.warning(
            "⚠️ Logical warning: Patient is under 16 and marked as married. Please verify the input."
        )

    bmi = weight / ((height / 100) ** 2)
    hypertension = 1 if hp == "Yes" else 0
    heart_disease = 1 if hd == "Yes" else 0

    input_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": float(age),
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": float(avg_glucose_level),
                "bmi": float(bmi),
                "smoking_status": smoking_status,
            }
        ]
    )

  
    y_prob = float(model.predict_proba(input_data)[:, 1][0])

    st.divider()
    st.subheader("Clinical Screening Result")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Calculated BMI", f"{bmi:.1f} kg/m²")
    m_col2.metric("Estimated Stroke Risk", f"{y_prob * 100:.1f}%")
    m_col3.metric("Screening Alert Threshold", f"{threshold * 100:.1f}%")

    if y_prob < threshold:
        st.success(
            f"✅ **Low/Moderate Risk Profile**: The estimated stroke probability ({y_prob * 100:.1f}%) is below the clinical screening threshold ({threshold * 100:.1f}%)."
        )
    else:
        st.error(
            f"🚨 **High Risk Profile Detected**: The estimated stroke probability ({y_prob * 100:.1f}%) exceeds the clinical screening threshold ({threshold * 100:.1f}%). Further medical evaluation is recommended."
        )