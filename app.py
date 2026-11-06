import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------
# 2. DESIGN / STYLING
# ----------------------------------

st.markdown(
    """
    <style>
        /* Main content spacing */
        .block-container {
            max-width: 1450px;
            padding-top: 3.5rem;
            padding-bottom: 3rem;
        }

        /* Headings */
        h1 {
            font-size: clamp(2rem, 3.2vw, 3.3rem) !important;
            font-weight: 750 !important;
            letter-spacing: -0.035em;
            line-height: 1.18 !important;
            margin-bottom: 0.65rem !important;
        }

        h2 {
            font-weight: 650 !important;
            letter-spacing: -0.02em;
        }

        h3 {
            font-weight: 600 !important;
        }

        /* Sidebar spacing */
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }

        section[data-testid="stSidebar"] h2 {
            font-size: 1.4rem !important;
        }

        /* Buttons */
        div.stButton > button,
        div.stFormSubmitButton > button {
            min-height: 3.1rem;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.65rem 1rem;
            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease;
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }

        /* Red primary buttons */
        button[kind="primary"] {
            background-color: #bf3650 !important;
            border-color: #bf3650 !important;
            color: #ffffff !important;
        }

        button[kind="primary"]:hover {
            background-color: #a92c44 !important;
            border-color: #a92c44 !important;
            color: #ffffff !important;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 1.15rem;
            height: 100%;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 700;
        }

        /* Tables */
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Notices */
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Chart images */
        div[data-testid="stImage"] {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Dividers */
        hr {
            margin-top: 1.8rem;
            margin-bottom: 1.8rem;
            opacity: 0.4;
        }

        /* Smaller screens */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            h1 {
                font-size: 2rem !important;
            }

            div[data-testid="stMetric"] {
                padding: 0.9rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------
# 3. LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "admission_model.joblib"


@st.cache_resource
def load_model(model_path, modified_time):
    # modified_time makes the cache refresh when the file changes.
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "admission_model.joblib is missing. "
        "Place it in the same folder as this script."
    )
    st.stop()


try:
    model_bundle = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime_ns,
    )

    model = model_bundle["model"]
    feature_names = list(model_bundle["features"])
    background = model_bundle["background"]

except Exception as exc:
    st.error(f"Model loading error: {exc}")
    st.stop()


# ----------------------------------
# 4. CREATE SHAP EXPLAINER
# ----------------------------------

@st.cache_resource
def create_explainer(_model, background, model_version):
    return shap.TreeExplainer(
        _model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )


try:
    explainer = create_explainer(
        model,
        background,
        MODEL_PATH.stat().st_mtime_ns,
    )

except Exception as exc:
    st.error(f"SHAP explainer error: {exc}")
    st.stop()


# ----------------------------------
# 5. HELPER FUNCTIONS
# ----------------------------------

def get_class_shap_values(values, class_index):
    """Handle list and array outputs from different SHAP versions."""
    if isinstance(values, list):
        return np.asarray(values[class_index])

    values = np.asarray(values)

    if values.ndim == 3:
        return values[:, :, class_index]

    return values


def get_base_value(expected_value, class_index):
    """Extract the expected value for the selected output."""
    expected_value = np.asarray(expected_value)

    if expected_value.ndim == 0:
        return float(expected_value)

    return float(expected_value[class_index])


def style_chart(fig):
    """Use transparent backgrounds and neutral chart labels."""
    label_color = "#888888"

    fig.patch.set_alpha(0)

    for ax in fig.axes:
        ax.set_facecolor("none")

        ax.title.set_color(label_color)
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)

        ax.tick_params(
            axis="both",
            colors=label_color,
            labelsize=10,
        )

        ax.xaxis.get_offset_text().set_color(label_color)
        ax.yaxis.get_offset_text().set_color(label_color)

        for spine in ax.spines.values():
            spine.set_color(label_color)
            spine.set_alpha(0.25)

        for text in ax.texts:
            original_color = str(text.get_color()).lower()

            # Preserve white labels inside SHAP's colored bars.
            if original_color not in {
                "white",
                "w",
                "#fff",
                "#ffffff",
            }:
                text.set_color(label_color)

        for label in (
            ax.get_xticklabels() + ax.get_yticklabels()
        ):
            label.set_color(label_color)

    return fig


def display_chart(fig):
    """Save the figure in memory and display it without st.pyplot."""
    try:
        with io.BytesIO() as buffer:
            fig.savefig(
                buffer,
                format="png",
                dpi=200,
                bbox_inches="tight",
                transparent=True,
            )
            image_bytes = buffer.getvalue()

        st.image(
            image_bytes,
            width="stretch",
        )

    finally:
        plt.close(fig)


# ----------------------------------
# 6. INITIALIZE INPUT STATE
# ----------------------------------

defaults = {
    "applicant_gpa": 3.50,
    "applicant_international": "Yes",
    "applicant_rating": 3,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ----------------------------------
# 7. SIDEBAR — APPLICANT DETAILS
# ----------------------------------

with st.sidebar:
    st.header("Applicant Details")

    st.caption(
        "Enter the applicant details below to generate a prediction."
    )

    st.divider()

    gpa = st.slider(
        "GPA Score",
        min_value=0.0,
        max_value=4.0,
        step=0.05,
        format="%.2f",
        key="applicant_gpa",
        help="Enter the applicant GPA on a 0–4 scale.",
    )

    st.write("")

    international = st.selectbox(
        "International Applicant?",
        ["Yes", "No"],
        key="applicant_international",
    )

    st.write("")

    university_rating = st.slider(
        "University Rating",
        min_value=1,
        max_value=5,
        key="applicant_rating",
    )

    st.divider()

    st.caption(
        "Research and educational use only. "
        "This tool does not provide official admission decisions."
    )


# ----------------------------------
# 8. MAIN PAGE HEADER
# ----------------------------------

st.title("Graduate Admission Predictor")

st.write(
    "Enter applicant details in the sidebar to estimate an admission "
    "outcome with Explainable AI (SHAP)."
)

st.caption(
    "Historical GradCafe records · Admission prediction · "
    "Individual and global explanations"
)

st.write("")


# ----------------------------------
# 9. CURRENT INPUT PARAMETERS
# ----------------------------------

st.header("Current Input Parameters")

preview = pd.DataFrame(
    {
        "GPA Score": [float(gpa)],
        "International Applicant": [international],
        "University Rating": [int(university_rating)],
    }
)

st.dataframe(
    preview,
    hide_index=True,
    width="stretch",
    column_config={
        "GPA Score": st.column_config.NumberColumn(
            "GPA Score",
            format="%.2f",
        ),
        "University Rating": st.column_config.NumberColumn(
            "University Rating",
            format="%d",
        ),
    },
)

submitted = st.button(
    "Predict & Generate XAI Analysis",
    type="primary",
)

st.caption(
    "Experimental estimates—not an official university "
    "decision or a guarantee."
)


# ----------------------------------
# 10. PREDICTION
# ----------------------------------

if submitted:
    try:
        values = {
            "GPA Score": float(gpa),
            "is_international": (
                1 if international == "Yes" else 0
            ),
            "University rating": float(university_rating),
        }

        missing_features = [
            name for name in feature_names if name not in values
        ]

        if missing_features:
            st.error(
                "The model requires features not supplied by this app: "
                + ", ".join(map(str, missing_features))
            )
            st.stop()

        row = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names,
        )

        classes = list(model.classes_)

        if len(classes) != 2 or 0 not in classes or 1 not in classes:
            st.error(
                "Expected a binary model with classes "
                "0 (rejected) and 1 (accepted)."
            )
            st.stop()

        accepted_index = classes.index(1)

        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        probability = float(probabilities[accepted_index])

        st.divider()
        st.header("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Acceptance Probability",
                f"{probability:.2%}",
            )

        with col2:
            st.metric(
                "Applicant GPA",
                f"{gpa:.2f} / 4.00",
            )

        with col3:
            st.metric(
                "University Rating",
                f"{university_rating} / 5",
            )

        st.write("")

        if prediction == 1:
            st.success("Predicted Outcome: ACCEPTED")
        else:
            st.warning("Predicted Outcome: REJECTED")

        st.progress(probability)

        st.caption(
            "The probability shown is the trained model's "
            "estimated probability for the accepted class."
        )

        display_names = {
            "GPA Score": "GPA Score",
            "is_international": "International Applicant",
            "University rating": "University Rating",
        }

        friendly_names = [
            display_names.get(name, name)
            for name in feature_names
        ]

        # ----------------------------------
        # 11. LOCAL XAI — INDIVIDUAL SHAP
        # ----------------------------------

        st.divider()
        st.header("Explainable AI — Individual Prediction")

        st.write(
            "This explanation shows how each applicant feature "
            "contributes to the model's estimated probability "
            "of acceptance."
        )

        with st.spinner("Generating individual explanation..."):
            local_values = explainer.shap_values(
                row,
                check_additivity=False,
            )

            local_values = get_class_shap_values(
                local_values,
                accepted_index,
            )

            base_value = get_base_value(
                explainer.expected_value,
                accepted_index,
            )

            if local_values.shape != (1, len(feature_names)):
                raise ValueError(
                    "Unexpected local SHAP shape: "
                    f"{local_values.shape}. Expected "
                    f"(1, {len(feature_names)})."
                )

            # Check that the explanation matches acceptance probability.
            explained_probability = (
                base_value + float(local_values[0].sum())
            )

            if not np.isclose(
                explained_probability,
                probability,
                atol=1e-3,
                rtol=1e-3,
            ):
                raise ValueError(
                    "The SHAP explanation does not match the "
                    "accepted-class probability. Check the model's "
                    "SHAP output format and class mapping."
                )

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
                feature_names=friendly_names,
            )

            fig1, ax1 = plt.subplots(figsize=(10, 5))

            try:
                shap.plots.waterfall(
                    explanation,
                    max_display=len(feature_names),
                    show=False,
                )

                waterfall_fig = plt.gcf()

                try:
                    style_chart(waterfall_fig)
                    waterfall_fig.tight_layout()

                    # No st.pyplot call: no savefig-kwargs warning.
                    display_chart(waterfall_fig)

                finally:
                    plt.close(waterfall_fig)

            finally:
                plt.close(fig1)

        st.caption(
            "Positive SHAP values push the prediction towards "
            "acceptance; negative values push it away from acceptance."
        )

        # ----------------------------------
        # 12. GLOBAL XAI
        # ----------------------------------

        st.divider()
        st.header("Global Feature Importance")

        st.write(
            "This chart shows the average absolute SHAP "
            "contribution of each feature across the background "
            "dataset."
        )

        with st.spinner("Calculating global feature importance..."):
            global_values = explainer.shap_values(
                background,
                check_additivity=False,
            )

            global_values = get_class_shap_values(
                global_values,
                accepted_index,
            )

            if (
                global_values.ndim != 2
                or global_values.shape[1] != len(feature_names)
            ):
                raise ValueError(
                    "Unexpected global SHAP shape: "
                    f"{global_values.shape}."
                )

            importance = pd.Series(
                np.abs(global_values).mean(axis=0),
                index=friendly_names,
            ).sort_values()

            fig2, ax2 = plt.subplots(figsize=(9, 4))

            try:
                importance.plot(
                    kind="barh",
                    ax=ax2,
                    width=0.6,
                )

                ax2.set_title(
                    "Mean Absolute SHAP Values — Accepted Class",
                    pad=16,
                )

                ax2.set_xlabel("Mean Absolute SHAP Value")
                ax2.set_ylabel("")

                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)

                ax2.grid(
                    axis="x",
                    linestyle="--",
                    alpha=0.15,
                    color="#888888",
                )

                ax2.set_axisbelow(True)

                style_chart(fig2)
                fig2.tight_layout()

                # Display the styled figure as an image.
                display_chart(fig2)

            finally:
                plt.close(fig2)

        st.subheader("Feature Contribution Summary")

        st.dataframe(
            importance.sort_values(ascending=False)
            .rename("Mean Absolute SHAP Value")
            .to_frame(),
            width="stretch",
        )

        # ----------------------------------
        # 13. DISCLAIMER
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
