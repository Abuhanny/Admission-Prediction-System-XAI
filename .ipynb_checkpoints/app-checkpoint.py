import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# =====================================
# LOAD TRAINED MODEL
# =====================================

model = joblib.load("admission_model.pkl")

# =====================================
# PAGE TITLE
# =====================================

st.title("🎓 University Admission Prediction System")

st.write("This app predicts whether a student may be admitted.")

# =====================================
# USER INPUTS
# =====================================

gre = st.slider("GRE Score", 260, 340, 300)

toefl = st.slider("TOEFL Score", 0, 120, 100)

university_rating = st.slider("University Rating", 1, 5, 3)

sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)

lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)

cgpa = st.slider("CGPA", 0.0, 10.0, 8.0)

research = st.selectbox("Research Experience", [0, 1])

# =====================================
# CREATE INPUT ARRAY
# =====================================

input_data = pd.DataFrame({
    "GRE Score": [gre],
    "TOEFL Score": [toefl],
    "University Rating": [university_rating],
    "SOP": [sop],
    "LOR": [lor],
    "CGPA": [cgpa],
    "Research": [research]
})

# =====================================
# PREDICTION BUTTON
# =====================================

if st.button("Predict Admission"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # =====================================
    # SHOW RESULT
    # =====================================

    if prediction == 1:
        st.success(f"✅ Admitted (Probability: {probability:.2f})")

    else:
        st.error(f"❌ Rejected (Probability: {probability:.2f})")

    # =====================================
    # SHAP EXPLANATION
    # =====================================

    st.subheader("📊 SHAP Feature Importance")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values[:, :, 1],
        input_data,
        show=False
    )

    st.pyplot(fig)

    # =====================================
    # LIME EXPLANATION
    # =====================================

    st.subheader("🔍 LIME Explanation")

    training_data = pd.read_csv("adm_data.csv")

    training_data.columns = training_data.columns.str.strip()

    training_data["Admitted"] = (
        training_data["Chance of Admit"] > 0.75
    ).astype(int)

    training_data = training_data.drop(
        ["Serial No.", "Chance of Admit"],
        axis=1,
        errors='ignore'
    )

    X = training_data.drop("Admitted", axis=1)

    lime_explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["Rejected", "Admitted"],
        mode="classification"
    )

    lime_exp = lime_explainer.explain_instance(
        input_data.iloc[0].values,
        model.predict_proba
    )

    st.components.v1.html(
        lime_exp.as_html(),
        height=800
    )