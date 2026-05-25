import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* APP BACKGROUND */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: white;
    color: black;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* TEXT */
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: black !important;
}

/* BUTTON */
.stButton button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: none;
    background-color: #2563EB;
    color: white;
    font-size: 16px;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #1D4ED8;
    color: white;
}

/* SLIDERS */
[data-baseweb="slider"] {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* TABS */
button[data-baseweb="tab"] {
    background-color: #E5E7EB !important;
    color: black !important;
    border-radius: 8px 8px 0px 0px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #2563EB !important;
    color: white !important;
}

/* METRIC */
[data-testid="stMetricValue"] {
    color: black;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("admission_model.pkl")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("adm_data.csv")

    df.columns = df.columns.str.strip()

    df["Admitted"] = (
        df["Chance of Admit"] > 0.75
    ).astype(int)

    df = df.drop(
        ["Serial No.", "Chance of Admit"],
        axis=1,
        errors="ignore"
    )

    return df

model = load_model()

data = load_data()

X_train = data.drop("Admitted", axis=1)

# =========================================================
# SHAP EXPLAINER
# =========================================================
shap_explainer = shap.TreeExplainer(model)

# =========================================================
# LIME EXPLAINER
# =========================================================
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Rejected", "Admitted"],
    mode="classification"
)

# =========================================================
# HEADER
# =========================================================
st.title("🎓 University Admission Predictor")

st.write(
    "Predict graduate school admission chances using "
    "Machine Learning + Explainable AI"
)

st.divider()

# =========================================================
# LAYOUT
# =========================================================
left, right = st.columns([1, 1])

# =========================================================
# LEFT PANEL - INPUTS
# =========================================================
with left:

    st.subheader("Student Information")

    gre = st.slider(
        "GRE Score",
        260,
        340,
        300
    )

    toefl = st.slider(
        "TOEFL Score",
        0,
        120,
        100
    )

    university_rating = st.slider(
        "University Rating",
        1,
        5,
        3
    )

    sop = st.slider(
        "Statement of Purpose Strength",
        1.0,
        5.0,
        3.0,
        step=0.5
    )

    lor = st.slider(
        "Letter of Recommendation Strength",
        1.0,
        5.0,
        3.0,
        step=0.5
    )

    cgpa = st.slider(
        "CGPA",
        0.0,
        10.0,
        8.0,
        step=0.1
    )

    research = st.selectbox(
        "Research Experience",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    predict = st.button(
        "Predict Admission"
    )

# =========================================================
# RIGHT PANEL - RESULTS
# =========================================================
with right:

    if predict:

        # -------------------------------------------------
        # INPUT DATAFRAME
        # -------------------------------------------------
        input_df = pd.DataFrame({
            "GRE Score": [gre],
            "TOEFL Score": [toefl],
            "University Rating": [university_rating],
            "SOP": [sop],
            "LOR": [lor],
            "CGPA": [cgpa],
            "Research": [research]
        })

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------
        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        percent = round(probability * 100, 2)

        admitted = prediction == 1

        # -------------------------------------------------
        # RESULT CARD
        # -------------------------------------------------
        with st.container(border=True):

            if admitted:

                st.success(
                    "✅ Likely Admitted"
                )

            else:

                st.error(
                    "❌ Likely Rejected"
                )

            st.metric(
                label="Admission Probability",
                value=f"{percent}%"
            )

            st.progress(float(probability))

        st.divider()

        # =================================================
        # TABS
        # =================================================
        tab1, tab2 = st.tabs([
            "📊 SHAP Visualization",
            "🔍 LIME Explanation"
        ])

        # =================================================
        # SHAP TAB
        # =================================================
        with tab1:

            st.subheader(
                "Feature Importance"
            )

            try:

                shap_values = shap_explainer(input_df)

                # Extract SHAP values
                shap_array = shap_values.values[0]

                # Fix multidimensional issue
                if len(shap_array.shape) > 1:
                    shap_array = shap_array[:, 1]

                # Convert to 1D
                shap_array = shap_array.flatten()

                # Create dataframe
                shap_df = pd.DataFrame({
                    "Feature": input_df.columns,
                    "Impact": shap_array
                })

                # Sort by impact
                shap_df = shap_df.sort_values(
                    by="Impact",
                    key=abs,
                    ascending=True
                )

                # Create plot
                fig, ax = plt.subplots(
                    figsize=(8, 4)
                )

                ax.barh(
                    shap_df["Feature"],
                    shap_df["Impact"]
                )

                ax.set_xlabel(
                    "SHAP Impact"
                )

                ax.set_ylabel(
                    "Feature"
                )

                ax.set_title(
                    "Feature Importance"
                )

                st.pyplot(fig)

                plt.close(fig)

                # SIMPLE EXPLANATION
                st.write("### Explanation")

                for _, row in shap_df.iterrows():

                    feature = row["Feature"]
                    impact = row["Impact"]

                    if impact > 0:

                        st.success(
                            f"{feature} increased the admission chance."
                        )

                    else:

                        st.error(
                            f"{feature} reduced the admission chance."
                        )

            except Exception as e:

                st.error(
                    f"SHAP visualization error: {e}"
                )

        # =================================================
        # LIME TAB
        # =================================================
        with tab2:

            st.subheader(
                "Why the Model Made This Prediction"
            )

            try:

                lime_result = lime_explainer.explain_instance(
                    input_df.iloc[0].values,
                    model.predict_proba,
                    num_features=7,
                    num_samples=100
                )

                lime_features = lime_result.as_list()

                # BAR CHART
                lime_df = pd.DataFrame(
                    lime_features,
                    columns=["Feature", "Weight"]
                )

                fig2, ax2 = plt.subplots(
                    figsize=(8, 4)
                )

                ax2.barh(
                    lime_df["Feature"],
                    lime_df["Weight"]
                )

                ax2.set_title(
                    "LIME Feature Contribution"
                )

                ax2.set_xlabel(
                    "Impact"
                )

                st.pyplot(fig2)

                plt.close(fig2)

                # SIMPLE TEXT EXPLANATION
                st.write("### Explanation")

                for feature, weight in lime_features:

                    if weight > 0:

                        st.success(
                            f"{feature} increases admission chance."
                        )

                    else:

                        st.error(
                            f"{feature} lowers admission chance."
                        )

            except Exception as e:

                st.error(
                    f"LIME explanation error: {e}"
                )

    else:

        st.info(
            "Fill the form and click Predict Admission"
        )