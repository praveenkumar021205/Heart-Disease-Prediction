
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CardioCare | Heart Disease Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
VISUALIZATION_DIR = BASE_DIR / "visualizations"

DATASET_PATH = DATA_DIR / "heart.csv"

LR_MODEL_PATH = MODEL_DIR / "logistic_regression_tuned.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest_tuned.pkl"

# ==========================================================
# 3. GLOBAL DASHBOARD THEME
# ==========================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */

    :root {
        --navy: #142C4E;
        --navy-light: #1D416D;
        --blue: #2878C8;
        --teal: #0C9A91;
        --background: #F3F6FB;
        --card: #FFFFFF;
        --text: #172B4D;
        --muted: #718096;
        --border: #E2EAF3;
    }

    .stApp {
        background-color: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(243, 246, 251, 0.95);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.4px;
    }

    p, li {
        line-height: 1.65;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #213F63;
    }

    section[data-testid="stSidebar"] > div {
        background: var(--navy);
    }

    section[data-testid="stSidebar"] * {
        color: #F4F8FF;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #345171;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 18px 0;
    }

    .brand-icon {
        background: #FFFFFF;
        color: var(--navy) !important;
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
    }

    .brand-title {
        color: #FFFFFF;
        font-size: 21px;
        font-weight: 750;
    }

    .brand-subtitle {
        color: #B9CCE3;
        font-size: 11px;
        margin-top: 3px;
    }

    .sidebar-section {
        color: #9DB6D3;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-size: 10px;
        font-weight: 700;
        margin: 14px 0 8px 0;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] {
        gap: 7px;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        border-radius: 9px;
        padding: 9px 12px;
        transition: background 0.2s ease;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #21436B;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 13px;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #2878C8;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF;
        font-weight: 700;
    }

    /* ---------- Page heading ---------- */

    .page-eyebrow {
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: 1.7px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .page-title {
        font-size: clamp(27px, 3vw, 36px);
        font-weight: 800;
        color: var(--navy);
        line-height: 1.2;
        margin-bottom: 9px;
    }

    .page-description {
        font-size: 14px;
        color: var(--muted);
        margin-bottom: 23px;
    }

    /* ---------- Cards ---------- */

    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px 19px;
        box-shadow: 0 3px 12px rgba(20, 44, 78, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 12px !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy) !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-radius: 15px;
        border-color: var(--border);
    }

    /* ---------- Buttons ---------- */

    .stButton > button,
    .stFormSubmitButton > button {
        background: var(--blue);
        color: #FFFFFF;
        border: 1px solid var(--blue);
        border-radius: 9px;
        padding: 10px 18px;
        font-weight: 650;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: var(--navy);
        border-color: var(--navy);
        color: #FFFFFF;
    }

    /* ---------- Inputs ---------- */

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 22px;
    }

    /* ---------- Data tables ---------- */

    [data-testid="stDataFrame"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Alerts ---------- */

    [data-testid="stAlert"] {
        border-radius: 11px;
    }

    /* ---------- Images ---------- */

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    /* ---------- Footer ---------- */

    .dashboard-footer {
        text-align: center;
        color: #7C8DA4;
        font-size: 11px;
        padding: 18px 0 4px 0;
    }

    /* ---------- Responsive ---------- */

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 22px !important;
        }

        .page-title {
            font-size: 27px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# 4. LOAD MODELS AND DATA
# ==========================================================

@st.cache_resource
def load_models():
    lr_model = joblib.load(LR_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)

    return lr_model, rf_model


@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)


try:
    lr_model, rf_model = load_models()
    df = load_data()

except Exception as error:
    st.error(
        "The application could not load the dataset or models. "
        "Please verify your project folder structure."
    )
    st.code(str(error))
    st.stop()


# ==========================================================
# 5. SHARED HELPER FUNCTIONS
# ==========================================================

def page_header(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="page-eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-description">{description}</div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, description=None):
    st.subheader(title)

    if description:
        st.caption(description)


def probability_of_positive(model, patient):
    """Return the probability of the positive class (target=1)."""

    probabilities = model.predict_proba(patient)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        return 0.0

    return float(probabilities[classes.index(1)])


def show_prediction_card(model_name, prediction, probability):
    with st.container(border=True):
        st.markdown(f"#### {model_name}")

        if prediction == 1:
            st.error("Positive prediction: heart disease indicated")
        else:
            st.success("Negative prediction: heart disease not indicated")

        st.metric(
            "Model-estimated positive-class probability",
            f"{probability:.2%}"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )


# ==========================================================
# 6. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">♥</div>
            <div>
                <div class="brand-title">CardioCare</div>
                <div class="brand-subtitle">MEDICAL ANALYTICS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-section">Main menu</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Patient Prediction",
            "Dataset Insights",
            "Model Comparison",
            "Explainable AI",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        """
        <div style="font-size:12px;color:#B9CCE3;line-height:1.8">
            <b>Heart Disease Prediction</b><br>
            Logistic Regression<br>
            Random Forest<br>
            SHAP Explainability
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("Academic project prototype")


# ==========================================================
# 7. GLOBAL APPLICATION HEADER
# ==========================================================

left, right = st.columns([3, 1])

with left:
    st.markdown(
        """
        <div style="font-size:12px;color:#718096">
            CARDIOCARE / MACHINE LEARNING PLATFORM
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        """
        <div style="text-align:right;font-size:12px;color:#718096">
            HEART HEALTH ANALYTICS
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# ==========================================================
# 8. HOME
# ==========================================================

if page == "Home":

    page_header(
        "Dashboard overview",
        "Heart Disease Analytics",
        "Monitor model performance, explore patient data, "
        "and review explainable machine learning results."
    )

    # Overview cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Records",
            f"{len(df):,}",
            help="Number of records in the loaded dataset."
        )

    with col2:
        st.metric(
            "Input Features",
            "13",
            help="The application uses 13 patient input features."
        )

    with col3:
        st.metric(
            "ML Models",
            "2",
            help="Logistic Regression and Random Forest."
        )

    with col4:
        st.metric(
            "Best Test ROC-AUC",
            "96.54%",
            help="Measured held-out test ROC-AUC for tuned Logistic Regression."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:

        with st.container(border=True):

            section_title(
                "Welcome to CardioCare",
                "An interactive machine learning project"
            )

            st.write(
                """
                This application explores the use of machine learning
                to classify heart disease status from selected clinical
                attributes in a dataset.

                It combines prediction, exploratory data analysis,
                model evaluation, and SHAP-based explanations in one
                interface.
                """
            )

            st.markdown("**Available capabilities**")

            st.markdown(
                """
                - Patient attribute-based predictions
                - Dataset exploration and visualization
                - Logistic Regression and Random Forest comparison
                - Model explainability using SHAP
                """
            )

            if st.button(
                "Open Patient Prediction",
                use_container_width=True
            ):
                st.session_state["go_to_prediction"] = True
                st.rerun()

    with right:

        with st.container(border=True):

            section_title(
                "Model performance",
                "Held-out test dataset"
            )

            model_names = [
                "Logistic Regression",
                "Random Forest"
            ]

            accuracies = [88.52, 88.52]

            fig, ax = plt.subplots(figsize=(6, 3.5))

            bars = ax.barh(
                model_names,
                accuracies,
                color=["#2878C8", "#0C9A91"],
                height=0.48
            )

            ax.set_xlim(0, 100)
            ax.set_xlabel("Accuracy (%)")
            ax.set_title("Test Accuracy")
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

            for bar, value in zip(bars, accuracies):
                ax.text(
                    value + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}%",
                    va="center",
                    fontsize=10
                )

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    section_title("Project workflow")

    flow_cols = st.columns(4)

    workflow = [
        ("01", "Input", "Enter patient attributes"),
        ("02", "Predict", "Run trained ML models"),
        ("03", "Compare", "Review model metrics"),
        ("04", "Explain", "Explore SHAP results")
    ]

    for col, item in zip(flow_cols, workflow):

        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size:11px;color:#0C9A91;"
                    f"font-weight:800'>{item[0]}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"#### {item[1]}")
                st.caption(item[2])

    st.info(
        "This academic prototype is not clinically validated and "
        "must not be used as a substitute for professional medical advice."
    )


# ==========================================================
# 9. PATIENT PREDICTION
# ==========================================================

elif page == "Patient Prediction":

    page_header(
        "Clinical input workspace",
        "Patient Prediction",
        "Enter the patient attributes to obtain predictions from "
        "the tuned Logistic Regression and Random Forest models."
    )

    with st.container(border=True):

        section_title(
            "Patient information",
            "Enter the available dataset features. "
            "Values should be consistent with the model's training data."
        )

        with st.form("patient_prediction_form"):

            st.markdown("##### Personal and clinical attributes")

            col1, col2, col3 = st.columns(3)

            with col1:

                age = st.number_input(
                    "Age (years)",
                    min_value=1,
                    max_value=120,
                    value=45
                )

                sex = st.selectbox(
                    "Sex",
                    options=[0, 1],
                    format_func=lambda x: (
                        "Female (0)" if x == 0 else "Male (1)"
                    )
                )

                cp = st.selectbox(
                    "Chest Pain Type (cp)",
                    options=[0, 1, 2, 3]
                )

                trestbps = st.number_input(
                    "Resting Blood Pressure",
                    min_value=50,
                    max_value=250,
                    value=120
                )

                chol = st.number_input(
                    "Serum Cholesterol",
                    min_value=50,
                    max_value=700,
                    value=200
                )

            with col2:

                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl",
                    options=[0, 1]
                )

                restecg = st.selectbox(
                    "Resting ECG Result",
                    options=[0, 1, 2]
                )

                thalach = st.number_input(
                    "Maximum Heart Rate",
                    min_value=50,
                    max_value=250,
                    value=150
                )

                exang = st.selectbox(
                    "Exercise-Induced Angina",
                    options=[0, 1]
                )

            with col3:

                oldpeak = st.number_input(
                    "ST Depression (oldpeak)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )

                slope = st.selectbox(
                    "ST Segment Slope",
                    options=[0, 1, 2]
                )

                ca = st.selectbox(
                    "Number of Major Vessels (ca)",
                    options=[0, 1, 2, 3, 4]
                )

                thal = st.selectbox(
                    "Thalassemia Result (thal)",
                    options=[0, 1, 2, 3]
                )

            submitted = st.form_submit_button(
                "Run Prediction",
                use_container_width=True
            )

    if submitted:

        patient_data = pd.DataFrame(
            [[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]],
            columns=[
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
            ]
        )

        try:

            lr_prediction = int(
                lr_model.predict(patient_data)[0]
            )

            rf_prediction = int(
                rf_model.predict(patient_data)[0]
            )

            lr_probability = probability_of_positive(
                lr_model,
                patient_data
            )

            rf_probability = probability_of_positive(
                rf_model,
                patient_data
            )

            st.markdown("<br>", unsafe_allow_html=True)

            section_title(
                "Prediction results",
                "Outputs from the two tuned models"
            )

            col1, col2 = st.columns(2)

            with col1:
                show_prediction_card(
                    "Logistic Regression",
                    lr_prediction,
                    lr_probability
                )

            with col2:
                show_prediction_card(
                    "Random Forest",
                    rf_prediction,
                    rf_probability
                )

            with st.container(border=True):

                section_title("Patient input summary")

                st.dataframe(
                    patient_data,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "The displayed probabilities are model-estimated "
                "positive-class probabilities. They are not calibrated "
                "clinical risk estimates."
            )

            st.warning(
                "This prediction is for academic demonstration only. "
                "It is not a diagnosis. Consult a qualified healthcare "
                "professional for medical assessment."
            )

        except Exception as error:

            st.error(
                "Prediction failed. Please verify the input features "
                "and confirm that the saved models are compatible."
            )

            st.code(str(error))

    else:

        st.info(
            "Complete the form and select Run Prediction to view "
            "the outputs of both models."
        )


# ==========================================================
# 10. DATASET INSIGHTS
# ==========================================================

elif page == "Dataset Insights":

    page_header(
        "Data exploration",
        "Dataset Insights",
        "Explore the dataset structure, missing values, feature "
        "distributions, target balance, and numerical correlations."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", f"{df.shape[1]}")

    with col3:
        st.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum())}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title(
            "Dataset preview",
            "First ten rows of the loaded CSV file"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title(
            "Data quality",
            "Data types and missing values per column"
        )

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(dtype) for dtype in df.dtypes
            ],
            "Missing Values": [
                int(value) for value in df.isna().sum()
            ]
        })

        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        with st.container(border=True):

            section_title("Target distribution")

            if "target" in df.columns:

                target_counts = (
                    df["target"]
                    .value_counts()
                    .sort_index()
                )

                target_counts.index = [
                    (
                        "No Heart Disease"
                        if value == 0
                        else "Heart Disease"
                        if value == 1
                        else str(value)
                    )
                    for value in target_counts.index
                ]

                fig, ax = plt.subplots(figsize=(6, 4))

                bars = ax.bar(
                    target_counts.index.astype(str),
                    target_counts.values,
                    color=["#2878C8", "#0C9A91"][
                        :len(target_counts)
                    ]
                )

                ax.set_ylabel("Number of Records")
                ax.set_title("Target Class Counts")
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)

                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{int(bar.get_height())}",
                        ha="center",
                        va="bottom"
                    )

                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:

                st.warning(
                    "A column named 'target' was not found in the dataset."
                )

    with right:

        with st.container(border=True):

            section_title("Feature distribution")

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:

                selected_feature = st.selectbox(
                    "Select a numerical feature",
                    numeric_columns
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                ax.hist(
                    df[selected_feature].dropna(),
                    bins=15,
                    color="#2878C8",
                    edgecolor="white"
                )

                ax.set_title(
                    f"Distribution of {selected_feature}"
                )

                ax.set_xlabel(selected_feature)
                ax.set_ylabel("Frequency")
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)

                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:

                st.info("No numerical features were found.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title(
            "Correlation analysis",
            "Pearson correlations between numerical columns"
        )

        numeric_df = df.select_dtypes(
            include="number"
        )

        if numeric_df.shape[1] >= 2:

            corr = numeric_df.corr()

            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            heatmap = ax.imshow(
                corr,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                aspect="auto"
            )

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(
                corr.columns,
                rotation=90
            )

            ax.set_yticklabels(corr.columns)

            fig.colorbar(
                heatmap,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                label="Correlation"
            )

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:

            st.info(
                "At least two numerical columns are required "
                "to create a correlation heatmap."
            )


# ==========================================================
# 11. MODEL COMPARISON
# ==========================================================

elif page == "Model Comparison":

    page_header(
        "Model evaluation",
        "Model Comparison",
        "Compare the tuned Logistic Regression and Random Forest "
        "models using the existing held-out test results."
    )

    # Previously measured results on the held-out test set.

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ],
        "Logistic Regression": [
            0.8852,
            0.8387,
            0.9286,
            0.8814,
            0.9654
        ],
        "Random Forest": [
            0.8852,
            0.8387,
            0.9286,
            0.8814,
            0.9513
        ]
    })

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Logistic Regression Accuracy",
            "88.52%"
        )

    with col2:
        st.metric(
            "Random Forest Accuracy",
            "88.52%"
        )

    with col3:
        st.metric(
            "Best ROC-AUC",
            "96.54%",
            help="Logistic Regression test ROC-AUC."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title(
            "Tuned model test results",
            "Metrics are displayed as percentages."
        )

        display_metrics = metrics.copy()

        for column in [
            "Logistic Regression",
            "Random Forest"
        ]:
            display_metrics[column] = display_metrics[
                column
            ].map(lambda value: f"{value:.2%}")

        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title(
            "Performance comparison",
            "Side-by-side comparison of all five evaluation metrics"
        )

        x = np.arange(len(metrics["Metric"]))
        width = 0.34

        lr_values = (
            metrics["Logistic Regression"].to_numpy() * 100
        )

        rf_values = (
            metrics["Random Forest"].to_numpy() * 100
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            x - width / 2,
            lr_values,
            width,
            label="Logistic Regression",
            color="#2878C8"
        )

        ax.bar(
            x + width / 2,
            rf_values,
            width,
            label="Random Forest",
            color="#0C9A91"
        )

        ax.set_ylabel("Score (%)")
        ax.set_title("Tuned Model Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(
            metrics["Metric"],
            rotation=15,
            ha="right"
        )

        ax.set_ylim(0, 105)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title(
            "Confusion matrix summary",
            "Classification counts on the held-out test dataset"
        )

        confusion_df = pd.DataFrame({
            "Classification Result": [
                "True Negative",
                "False Positive",
                "False Negative",
                "True Positive"
            ],
            "Logistic Regression": [
                28, 5, 2, 26
            ],
            "Random Forest": [
                28, 5, 2, 26
            ]
        })

        st.dataframe(
            confusion_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Both tuned models produced the same class predictions "
            "on this test set. Their ROC-AUC scores differ."
        )

    st.info(
        "Both models achieved 88.52% test accuracy. Logistic Regression "
        "had a higher ROC-AUC (96.54%) than Random Forest (95.13%)."
    )

    st.caption(
        "These figures are the previously measured results on the "
        "61-record held-out test set. They are not newly calculated "
        "during application execution and do not establish clinical validity."
    )


# ==========================================================
# 12. EXPLAINABLE AI
# ==========================================================

elif page == "Explainable AI":

    page_header(
        "Model transparency",
        "Explainable AI",
        "Explore SHAP visualizations that help describe how "
        "features contribute to the trained model's outputs."
    )

    st.markdown(
        """
        **SHAP** stands for SHapley Additive exPlanations. It is a
        method for explaining model predictions by attributing
        contributions to input features.
        """
    )

    importance_path = (
        VISUALIZATION_DIR / "shap_feature_importance.png"
    )

    patient_path = (
        VISUALIZATION_DIR / "shap_patient_explanation.png"
    )

    with st.container(border=True):

        section_title(
            "Global feature importance",
            "Overview of feature contributions in the saved explanation."
        )

        if importance_path.exists():

            st.image(
                str(importance_path),
                use_container_width=True
            )

        else:

            st.warning(
                "Global SHAP plot not found. "
                "Run src/explain_model.py to generate it."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title(
            "Example patient explanation",
            "Saved explanation for an example test-set record."
        )

        if patient_path.exists():

            st.image(
                str(patient_path),
                use_container_width=True
            )

        else:

            st.warning(
                "Patient SHAP plot not found. "
                "Run src/explain_model.py to generate it."
            )

    st.info(
        "The displayed patient explanation is a saved test-sample "
        "example. It does not explain the patient currently entered "
        "on the Patient Prediction page."
    )

    st.caption(
        "SHAP explains model behavior; it does not establish causation, "
        "clinical reliability, or an individual medical diagnosis."
    )


# ==========================================================
# 13. ABOUT
# ==========================================================

elif page == "About":

    page_header(
        "Project information",
        "About CardioCare",
        "An academic machine learning project for heart disease "
        "classification and model explainability."
    )

    with st.container(border=True):

        section_title("Project title")

        st.markdown(
            """
            ### Heart Disease Prediction and Explainable Risk Assessment

            **Algorithms:** Logistic Regression and Random Forest
            """
        )

        st.write(
            """
            The application demonstrates how a trained classification
            pipeline can accept patient attributes, generate model
            predictions, compare evaluation results, and display SHAP
            explanations.
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            section_title("Dataset")

            st.write(
                """
                The project uses the Cleveland Heart Disease dataset
                from the UCI Machine Learning Repository.
                """
            )

            st.markdown(
                """
                - 303 records
                - 13 input features
                - Binary classification target
                """
            )

            st.caption(
                "Target 0 indicates no heart disease in the dataset; "
                "target 1 indicates heart disease in the dataset."
            )

    with col2:

        with st.container(border=True):

            section_title("Technologies")

            st.markdown(
                """
                - Python
                - Pandas and NumPy
                - Scikit-learn
                - Matplotlib
                - Streamlit
                - SHAP
                - Joblib
                """
            )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        section_title("Project workflow")

        workflow = [
            "Load the dataset",
            "Preprocess numerical and categorical features",
            "Train Logistic Regression and Random Forest",
            "Tune model hyperparameters",
            "Evaluate held-out test performance",
            "Generate SHAP visualizations",
            "Present results through Streamlit"
        ]

        for index, item in enumerate(workflow, start=1):

            st.markdown(
                f"**{index:02d}.** {item}"
            )

    st.warning(
        "Medical disclaimer: This is an academic prototype and has "
        "not been clinically validated. It must not replace professional "
        "medical advice, diagnosis, or treatment."
    )


# ==========================================================
# 14. SHARED FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div class="dashboard-footer">
        CardioCare Analytics · Heart Disease Prediction Project<br>
        Academic Prototype · Not for clinical diagnosis
    </div>
    """,
    unsafe_allow_html=True
)
