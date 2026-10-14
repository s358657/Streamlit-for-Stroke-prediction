import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊", layout="wide")


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


df = load_data("healthcare-dataset-stroke-data.csv")

st.title("📊 Exploratory Data Analysis of the Dataset")


st.subheader("1. Class Imbalance")
stroke_pct = df["stroke"].value_counts(normalize=True) * 100
col1, col2 = st.columns(2)
col1.metric("Patients who had a stroke", f"{stroke_pct[1]:.2f}%")
col2.metric("Patients who did not have a stroke", f"{stroke_pct[0]:.2f}%")
st.info(
    "Due to the severe class imbalance in the dataset (~4.87% positive cases), standard accuracy is misleading; recall-oriented metrics like the F2-score and PR-AUC are required."
)


st.subheader("2. Stroke Incidence Rate by Age Group")
df_age = df.copy()
bins_age = [0, 20, 30, 40, 50, 60, 70, 80, 90]
labels_age = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "80+"]
df_age["age_group"] = pd.cut(df_age["age"], bins=bins_age, labels=labels_age)

age_incidence = df_age.groupby("age_group", observed=False)["stroke"].mean() * 100
st.bar_chart(age_incidence, y_label="Stroke Incidence Rate (%)", x_label="Age Group")
st.info(
    "By normalizing for group size, we observe that the probability of experiencing a stroke increases exponentially after age 50, exceeding 15-20% in patients over 70."
)


st.subheader("3. Impact of Hypertension and Heart Disease")
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Hypertension Risk")
    hyper_data = (
        df.groupby("hypertension")["stroke"].value_counts(normalize=True).unstack()
        * 100
    )
    hyper_data.index = ["Non-Hypertensive", "Hypertensive"]
    hyper_data = hyper_data.rename(columns={0: "No Stroke", 1: "Stroke"})

    st.bar_chart(hyper_data["Stroke"], y_label="Stroke Incidence (%)", color="#d62728")
    abs_diff_hyper = (
        hyper_data.loc["Hypertensive", "Stroke"]
        - hyper_data.loc["Non-Hypertensive", "Stroke"]
    )
    rel_risk_hyper = (
        hyper_data.loc["Hypertensive", "Stroke"]
        / hyper_data.loc["Non-Hypertensive", "Stroke"]
    )
    st.metric(
        "Stroke incidence in hypertensive patients",
        f"{hyper_data.loc['Hypertensive', 'Stroke']:.2f}%",
        delta=f"+{abs_diff_hyper:.2f} percentage points ({rel_risk_hyper:.1f}x relative risk)",
        delta_color="inverse",
    )

with col4:
    st.markdown("#### Heart Disease Risk")
    heart_data = (
        df.groupby("heart_disease")["stroke"].value_counts(normalize=True).unstack()
        * 100
    )
    heart_data.index = ["No Heart Disease", "Heart Disease"]
    heart_data = heart_data.rename(columns={0: "No Stroke", 1: "Stroke"})

    st.bar_chart(heart_data["Stroke"], y_label="Stroke Incidence (%)", color="#d62728")
    abs_diff_heart = (
        heart_data.loc["Heart Disease", "Stroke"]
        - heart_data.loc["No Heart Disease", "Stroke"]
    )
    rel_risk_heart = (
        heart_data.loc["Heart Disease", "Stroke"]
        / heart_data.loc["No Heart Disease", "Stroke"]
    )
    st.metric(
        "Stroke incidence in heart disease patients",
        f"{heart_data.loc['Heart Disease', 'Stroke']:.2f}%",
        delta=f"+{abs_diff_heart:.2f} percentage points ({rel_risk_heart:.1f}x relative risk)",
        delta_color="inverse",
    )

st.info(
    "Patients with hypertension or heart disease exhibit more than 3 times the relative risk of experiencing a stroke compared to patients without these conditions."
)


st.subheader("4. BMI Category and Glucose Level by Stroke Status")
df_clean = df.dropna(subset=["bmi"]).copy()
bins_bmi = [0, 18.5, 24.9, 29.9, 100]
labels_bmi = ["Underweight", "Normal weight", "Overweight", "Obesity"]
df_clean["bmi_category"] = pd.cut(df_clean["bmi"], bins=bins_bmi, labels=labels_bmi)
df_clean["Stroke Status"] = df_clean["stroke"].map({0: "No Stroke", 1: "Stroke"})

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(
    data=df_clean,
    x="bmi_category",
    y="avg_glucose_level",
    hue="Stroke Status",
    palette={"No Stroke": "#8dd3c7", "Stroke": "#fb8072"},
    ax=ax,
)
ax.set_xlabel("BMI Category")
ax.set_ylabel("Average Glucose Level (mg/dL)")
st.pyplot(fig)
st.info(
    "💡 **Clinical Insight:** Across normal, overweight, and obese categories, patients who suffered a stroke (red boxes) show a markedly higher median glucose level and upper quartile compared to non-stroke patients, confirming the role of hyperglycemia and metabolic factors."
)


st.subheader("5. Stroke Incidence Rate by Gender")
df_gender = df[df["gender"].isin(["Male", "Female"])].copy()
gender_incidence = df_gender.groupby("gender")["stroke"].mean() * 100

col5, col6 = st.columns(2)
col5.metric("Stroke Incidence in Males", f"{gender_incidence['Male']:.2f}%")
col6.metric("Stroke Incidence in Females", f"{gender_incidence['Female']:.2f}%")
st.info(
    "💡 **Statistical Note:** While the raw number of strokes is higher in females due to a larger female sample size in the dataset (~59%), the normalized incidence rate is actually slightly higher in males (5.11% vs. 4.71%)."
)