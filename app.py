
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Graduate Admission Predictor")

st.write(
    "Predict graduate admission outcomes using historical "
    "GradCafe admissions records, with Explainable AI (SHAP)."
)

st.info(
    "This is a research/educational estimate, not an official "
    "university decision or a guarantee."
)

# ----------------------------------
# 2. LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = Path(__file__).parent / "admission_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("admission_model.joblib is missing.")
    st.stop()

try:
    model_bundle = load_model()

    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    background = model_bundle["background"]

except Exception as exc:
    st.error(f"Model loading error: {exc}")
    st.stop()


# ----------------------------------
# 3. CREATE SHAP EXPLAINER
# ----------------------------------

@st.cache_resource
def create_explainer(_model, background):
    return shap.TreeExplainer(
        _model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability"
    )


try:
    explainer = create_explainer(model, background)

except Exception as exc:
    st.error(f"SHAP explainer error: {exc}")
    st.stop()


# ----------------------------------
# 4. HELPER FUNCTIONS
# ----------------------------------

def get_class_shap_values(values, class_index):

    # Older SHAP versions return a list
    if isinstance(values, list):
        return np.asarray(values[class_index])

    values = np.asarray(values)

    # Newer SHAP versions may return 3D arrays
    if values.ndim == 3:
        return values[:, :, class_index]

    return values


def get_base_value(expected_value, class_index):

    expected_value = np.asarray(expected_value)

    if expected_value.ndim == 0:
        return float(expected_value)

    return float(expected_value[class_index])


# ----------------------------------
# 5. USER INPUT
# ----------------------------------

st.header("Applicant Information")

with st.form("prediction_form"):

    gpa = st.number_input(
        "GPA Score",
        min_value=0.0,
        max_value=4.0,
        value=3.50,
        step=0.05
    )

    international = st.selectbox(
        "International Applicant?",
        ["Yes", "No"]
    )

    university_rating = st.slider(
        "University Rating",
        min_value=1,
        max_value=5,
        value=3
    )

    submitted = st.form_submit_button(
        "Predict Admission Outcome",
        type="primary"
    )


# ----------------------------------
# 6. PREDICTION
# ----------------------------------

if submitted:

    values = {
        "GPA Score": float(gpa),
        "is_international": (
            1 if international == "Yes" else 0
        ),
        "University rating": float(university_rating)
    }

    row = pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names
    )

    try:

        prediction = int(model.predict(row)[0])

        classes = list(model.classes_)

        if 1 not in classes or 0 not in classes:
            st.error("Expected model classes 0 and 1.")
            st.stop()

        accepted_index = classes.index(1)

        probabilities = model.predict_proba(row)[0]

        probability = float(
            probabilities[accepted_index]
        )

        st.divider()
        st.header("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Acceptance Probability",
                f"{probability:.2%}"
            )

        with col2:
            if prediction == 1:
                st.success("Predicted Outcome: ACCEPTED")
            else:
                st.warning("Predicted Outcome: REJECTED")

        st.progress(probability)

        # ----------------------------------
        # 7. LOCAL XAI - INDIVIDUAL SHAP
        # ----------------------------------

       
        # ----------------------------------
        # 7. LOCAL XAI - INDIVIDUAL SHAP
        # ----------------------------------

        st.divider()

        st.header("🔍 Explainable AI - Individual Prediction")

        st.write(
            "This explanation shows how each applicant feature "
            "contributes to the model's estimated probability "
            "of acceptance."
        )

        local_values = explainer.shap_values(
            row,
            check_additivity=False
        )

        local_values = get_class_shap_values(
            local_values,
            accepted_index
        )

        base_value = get_base_value(
            explainer.expected_value,
            accepted_index
        )

        display_names = {
            "GPA Score": "GPA Score",
            "is_international": "International Applicant",
            "University rating": "University Rating"
        }

        friendly_names = [
            display_names.get(name, name)
            for name in feature_names
        ]

        display_values = row.iloc[0].tolist()

        for i, name in enumerate(feature_names):
            if name == "is_international":
                display_values[i] = (
                    "Yes" if display_values[i] == 1 else "No"
                )

        explanation = shap.Explanation(
            values=local_values[0],
            base_values=base_value,
            data=display_values,
            feature_names=friendly_names
        )

        fig1, ax1 = plt.subplots(figsize=(10, 5))

        shap.plots.waterfall(
            explanation,
            max_display=len(feature_names),
            show=False
        )

        st.pyplot(fig1)
        plt.close(fig1)

        st.caption(
            "Positive SHAP values push the prediction towards "
            "acceptance; negative values push it away from acceptance."
        )


        # ----------------------------------
        # 8. GLOBAL XAI
        # ----------------------------------

        st.divider()

        st.header("🌍 Global Feature Importance")

        st.write(
            "This chart shows the average absolute SHAP "
            "contribution of each feature across the background "
            "dataset."
        )

        global_values = explainer.shap_values(
            background,
            check_additivity=False
        )

        global_values = get_class_shap_values(
            global_values,
            accepted_index
        )

        importance = pd.Series(
    np.abs(global_values).mean(axis=0),
    index=friendly_names
).sort_values()

        fig2, ax2 = plt.subplots(figsize=(9, 4))

        importance.plot(
            kind="barh",
            ax=ax2
        )

        ax2.set_title(
            "Mean Absolute SHAP Values - Accepted Class"
        )

        ax2.set_xlabel(
            "Mean Absolute SHAP Value"
        )

        ax2.set_ylabel("Features")

        fig2.tight_layout()

        st.pyplot(fig2)
        plt.close(fig2)

        st.write("### Feature contribution summary")

        st.dataframe(
            importance.sort_values(ascending=False)
            .rename("Mean Absolute SHAP Value")
            .to_frame(),
            use_container_width=True
        )

        # ----------------------------------
        # 9. DISCLAIMER
        # ----------------------------------

        st.divider()

        st.caption(
            "SHAP explains the behavior of the trained model, "
            "not causal effects on actual admission decisions. "
            "Historical GradCafe reports may be self-selected. "
            "Predictions are experimental estimates, not guarantees."
        )

    except Exception as exc:

        st.error(
            f"Prediction or SHAP explanation failed: {exc}"
        )

