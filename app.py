import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components


# ============================================================
# Warnings
# ============================================================

# These warnings come from loading serialized sklearn/xgboost
# artifacts created with older versions.
# They are handled separately in the deployment environment.
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*serialized model.*older version of XGBoost.*"
)


# ============================================================
# App Configuration
# ============================================================

st.set_page_config(
    page_title="🦇 LA Crime Explorer",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="auto"
)


# ============================================================
# Project Paths
# ============================================================

FILES_DIR = Path("Files")
DATA_FILE = Path("crime_sample.csv")

SAMPLE_SIZE = 50000


# ============================================================
# Model Configuration
# ============================================================

MODEL_NAMES = [
    "XGBoost",
    "RandomForestClassifier",
    "LogisticRegression",
]

MODEL_DISPLAY_NAMES = {
    "XGBoost": "XGBoost",
    "RandomForestClassifier": "Random Forest",
    "LogisticRegression": "Logistic Regression",
}


# ============================================================
# Batman Theme
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --yellow: #f1c40f;
        --yellow-dark: #d4ac0d;
        --bg: #0d0d0d;
        --panel: #181818;
        --border: #333333;
        --text: #f5f5f5;
        --muted: #bdbdbd;
    }

    html,
    body,
    .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1,
    h2,
    h3,
    h4 {
        color: var(--yellow) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--panel) !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--yellow) !important;
    }

    div.stButton > button,
    .stFormSubmitButton > button {
        background-color: var(--yellow) !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }

    div.stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: var(--yellow-dark) !important;
        color: #000000 !important;
    }

    a,
    a:link,
    a:visited {
        color: var(--yellow) !important;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--yellow) !important;
        font-weight: 800 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

@st.cache_resource
def load_pickle(filename):
    path = FILES_DIR / filename

    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        st.error(
            f"Could not load `{filename}`.\n\n"
            f"Error: {e}"
        )
        return None


@st.cache_data
def load_data(sample_size=SAMPLE_SIZE):

    if not DATA_FILE.exists():
        return None

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    # Avoid pandas date inference warning.
    if "DATE OCC" in df.columns:
        df["DATE OCC"] = pd.to_datetime(
            df["DATE OCC"],
            errors="coerce",
            format="mixed"
        )

    if sample_size and len(df) > sample_size:
        df = df.sample(
            n=sample_size,
            random_state=42
        )

    return df


def get_model_accuracy(results, model_name):

    if not results:
        return None

    if model_name not in results:
        return None

    result = results[model_name]

    if not isinstance(result, dict):
        return None

    metrics = result.get("metrics", {})

    possible_keys = [
        "accuracy",
        "Accuracy",
        "acc",
        "ACC"
    ]

    for key in possible_keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return None

    return None


def load_results():

    path = FILES_DIR / "results_dict.pkl"

    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)

    except Exception:
        return None


def get_ordered_models(results=None):

    available = [
        model
        for model in MODEL_NAMES
        if model in available_models
    ]

    if not results:
        return available

    accuracies = {
        model: get_model_accuracy(
            results,
            model
        )
        for model in available
    }

    # Sort by accuracy descending.
    # If accuracy is unavailable, keep the preferred order.
    ordered = sorted(
        available,
        key=lambda model: (
            accuracies[model] is not None,
            accuracies[model] if accuracies[model] is not None else -1
        ),
        reverse=True
    )

    return ordered


# ============================================================
# Load Preprocessing Artifacts
# ============================================================

le_crm_cd_desc = load_pickle(
    "Crm Cd Desc_label_encoder.pkl"
)

le_vict_descent = load_pickle(
    "Vict Descent_label_encoder.pkl"
)

le_weapon_desc = load_pickle(
    "Weapon Desc_label_encoder.pkl"
)

le_status_desc = load_pickle(
    "Status Desc_label_encoder.pkl"
)

le_y = load_pickle(
    "le_y.pkl"
)

yeo_johnson = load_pickle(
    "yeo-johnson.pkl"
)

std_scaler = load_pickle(
    "StandardScaler.pkl"
)

std2_scaler = load_pickle(
    "StandardScaler2.pkl"
)

robust_scaler = load_pickle(
    "RobustScaler.pkl"
)


# ============================================================
# Available Models
# ============================================================

available_models = {}

for model_name in MODEL_NAMES:

    model_path = FILES_DIR / f"{model_name}.pkl"

    if model_path.exists():
        available_models[model_name] = str(model_path)


# ============================================================
# Victim Sex Mapping
# ============================================================

VICT_SEX_MAP = {
    "M": 0,
    "F": 1,
    "H": 2,
    "X": 3
}


# ============================================================
# Model Features
# ============================================================

MODEL_FEATURES = [
    "Crm Cd 1",
    "Crm Cd Desc",
    "Weapon Used Cd",
    "Premis Cd",
    "Weapon Desc",
    "Vict Age",
    "Vict Descent",
    "Vict Sex",
    "Status Desc"
]


# ============================================================
# Preprocessing
# ============================================================

def encode_and_transform(input_df):

    df = input_df.copy()

    required_encoders = {
        "Crm Cd Desc": le_crm_cd_desc,
        "Vict Descent": le_vict_descent,
        "Weapon Desc": le_weapon_desc,
        "Status Desc": le_status_desc,
    }

    for name, encoder in required_encoders.items():

        if encoder is None:
            raise ValueError(
                f"{name} label encoder was not found."
            )

    try:

        df["Crm Cd Desc"] = (
            le_crm_cd_desc.transform(
                df["Crm Cd Desc"].astype(str)
            )
        )

        df["Vict Descent"] = (
            le_vict_descent.transform(
                df["Vict Descent"].astype(str)
            )
        )

        df["Weapon Desc"] = (
            le_weapon_desc.transform(
                df["Weapon Desc"].astype(str)
            )
        )

        df["Status Desc"] = (
            le_status_desc.transform(
                df["Status Desc"].astype(str)
            )
        )

    except ValueError as e:

        raise ValueError(
            "One or more categorical values were not "
            "seen during model training.\n\n"
            f"Details: {e}"
        )

    # Victim sex
    df["Vict Sex"] = df["Vict Sex"].map(
        lambda value:
        VICT_SEX_MAP.get(value, value)
        if isinstance(value, str)
        else value
    )

    df["Vict Sex"] = pd.to_numeric(
        df["Vict Sex"],
        errors="coerce"
    ).fillna(0)

    # Standard scaling
    if std_scaler is not None:

        df[
            ["Crm Cd Desc", "Vict Descent"]
        ] = std_scaler.transform(
            df[
                ["Crm Cd Desc", "Vict Descent"]
            ]
        )

    # Log transform
    for column in [
        "Crm Cd 1",
        "Premis Cd"
    ]:

        df[column] = np.log1p(
            pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)
        )

    # Standard scaler 2
    if std2_scaler is not None:

        df[
            ["Crm Cd 1", "Premis Cd"]
        ] = std2_scaler.transform(
            df[
                ["Crm Cd 1", "Premis Cd"]
            ]
        )

    # Numeric features
    for column in [
        "Vict Age",
        "Weapon Used Cd"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    # Robust scaler
    if robust_scaler is not None:

        df[
            ["Vict Age", "Weapon Used Cd"]
        ] = robust_scaler.transform(
            df[
                ["Vict Age", "Weapon Used Cd"]
            ]
        )

    # Yeo-Johnson
    if yeo_johnson is not None:

        columns = [
            "Weapon Desc",
            "Status Desc",
            "Vict Sex"
        ]

        df[columns] = yeo_johnson.transform(
            df[columns]
        )

    return df[MODEL_FEATURES]


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🦇 Navigator")

page = st.sidebar.selectbox(
    "Navigate",
    [
        "Home",
        "Predict",
        "Visualizations",
        "Model Comparison",
        "About"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title("🦇 Los Angeles Crime Explorer")

    st.markdown(
        """
        Explore crime patterns across Los Angeles,
        discover the most common incidents,
        and use machine learning to predict whether
        an incident belongs to **Part 1 or Part 2**.
        """
    )

    st.divider()

    df_home = load_data()

    if df_home is None:

        st.error(
            f"Dataset `{DATA_FILE}` was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_crimes = len(df_home)

    unique_crimes = (
        df_home["Crm Cd Desc"].nunique()
        if "Crm Cd Desc" in df_home.columns
        else 0
    )

    part1_percentage = 0

    if "Part 1-2" in df_home.columns:

        part1_count = (
            df_home["Part 1-2"] == 1
        ).sum()

        if total_crimes > 0:
            part1_percentage = (
                part1_count /
                total_crimes *
                100
            )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sampled Incidents",
            f"{total_crimes:,}"
        )

    with col2:
        st.metric(
            "Crime Types",
            f"{unique_crimes:,}"
        )

    with col3:
        st.metric(
            "Deployed Models",
            len(available_models)
        )

    with col4:
        st.metric(
            "Part 1 Share",
            f"{part1_percentage:.1f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # Crime Trend
    # --------------------------------------------------------

    st.subheader("📈 Crime Trend")

    if "DATE OCC" in df_home.columns:

        monthly = (
            df_home
            .dropna(subset=["DATE OCC"])
            .assign(
                year_month=(
                    df_home["DATE OCC"]
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )
            )
            .groupby("year_month")
            .size()
            .reset_index(name="count")
        )

        fig = px.line(
            monthly,
            x="year_month",
            y="count",
            markers=True,
            labels={
                "year_month": "Month",
                "count": "Number of Incidents"
            }
        )

        fig.update_traces(
            line=dict(
                color="#f1c40f",
                width=3
            ),
            marker=dict(size=6)
        )

        fig.update_layout(
            height=390,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="#f5f5f5"),
            xaxis=dict(gridcolor="#2c2c2c"),
            yaxis=dict(gridcolor="#2c2c2c")
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # Top Crimes
    # --------------------------------------------------------

    if "Crm Cd Desc" in df_home.columns:

        st.subheader("Most Common Crime Types")

        top_crimes = (
            df_home["Crm Cd Desc"]
            .value_counts()
            .nlargest(10)
            .sort_values()
        )

        fig2 = px.bar(
            x=top_crimes.values,
            y=top_crimes.index,
            orientation="h",
            labels={
                "x": "Number of Incidents",
                "y": ""
            }
        )

        fig2.update_traces(
            marker_color="#e67e22"
        )

        fig2.update_layout(
            height=430,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="#f5f5f5"),
            xaxis=dict(gridcolor="#2c2c2c"),
            yaxis=dict(gridcolor="#2c2c2c")
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )

    st.divider()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    st.subheader("🦇 Explore Gotham")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:

        with st.container(border=True):

            st.markdown("### Predict Crime Category")

            st.write(
                "Enter details and use the "
                "trained ML models to predict whether "
                "the crime is classified as Part 1 or Part 2."
            )

    with feature2:

        with st.container(border=True):

            st.markdown("### Explore Crime Analytics")

            st.write(
                "Analyze crime trends, time patterns, "
                "crime locations, and the most common "
                "crime types across Los Angeles."
            )

    with feature3:

        with st.container(border=True):

            st.markdown("### Compare ML Models")

            st.write(
                "Inspect model accuracy, metrics, "
                "confusion matrices, and classification reports."
            )

    if "DATE OCC" in df_home.columns:

        valid_dates = df_home["DATE OCC"].dropna()

        if not valid_dates.empty:

            min_date = valid_dates.min()
            max_date = valid_dates.max()

            date_range = (
                f"{min_date.strftime('%Y-%m-%d')} "
                f"→ "
                f"{max_date.strftime('%Y-%m-%d')}"
            )

            st.caption(
                f"Dataset: LAPD Crime Data • "
                f"Deployment sample: {total_crimes:,} incidents • "
                f"Period: {date_range}"
            )


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict":

    st.title("Predict Crime Category")

    st.write(
        "Use a trained machine learning model to predict "
        "whether the incident belongs to **Part 1 or Part 2**."
    )

    # --------------------------------------------------------
    # Explain Target Clearly
    # --------------------------------------------------------

    st.info(
        """
        **What does Part 1 vs Part 2 mean?**

        **Part 1** → Major/serious crime category.

        **Part 2** → Other, generally less serious offenses.
        """
    )

    # --------------------------------------------------------
    # Load Results
    # --------------------------------------------------------

    prediction_results = load_results()

    ordered_models = get_ordered_models(
        prediction_results
    )

    if not ordered_models:

        st.error(
            "No trained models were found. "
            "Check the Files/ folder."
        )

        st.stop()

    # --------------------------------------------------------
    # Model Selection
    # --------------------------------------------------------

    model_choice = st.selectbox(
        "Choose model for prediction",
        options=ordered_models,
        index=0,
        format_func=lambda x:
        MODEL_DISPLAY_NAMES.get(x, x)
    )

    if prediction_results:

        selected_accuracy = get_model_accuracy(
            prediction_results,
            model_choice
        )

        if selected_accuracy is not None:

            st.caption(
                f"Validation accuracy: "
                f"{selected_accuracy:.2%}"
            )

    model = load_pickle(
        f"{model_choice}.pkl"
    )

    if model is None:

        st.error(
            "Could not load the selected model."
        )

        st.stop()

    if le_y is None:

        st.error(
            "`le_y.pkl` was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # Form
    # --------------------------------------------------------

    with st.form("predict_form"):

        st.subheader("Incident Features")

        col1, col2, col3 = st.columns(3)

        with col1:

            crm_cd1 = st.number_input(
                "Crm Cd 1",
                value=200.0,
                step=1.0
            )

            crm_cd_desc = st.text_input(
                "Crm Cd Desc",
                value="THEFT OF IDENTITY"
            )

            weapon_used_cd = st.number_input(
                "Weapon Used Cd",
                value=0.0,
                step=1.0
            )

        with col2:

            premis_cd = st.number_input(
                "Premis Cd",
                value=400.0,
                step=1.0
            )

            weapon_desc = st.text_input(
                "Weapon Desc",
                value="No Weapon"
            )

            vict_age = st.number_input(
                "Victim Age",
                value=30.0,
                min_value=0.0,
                step=1.0
            )

        with col3:

            vict_descent = st.selectbox(
                "Victim Descent",
                [
                    "H",
                    "W",
                    "B",
                    "O",
                    "A",
                    "I",
                    "U",
                    "X",
                    "Z"
                ]
            )

            vict_sex = st.selectbox(
                "Victim Sex",
                [
                    "M",
                    "F",
                    "H",
                    "X"
                ]
            )

            status_desc = st.text_input(
                "Status Desc",
                value="Invest Cont"
            )

        submit = st.form_submit_button(
            "Predict"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if submit:

        sample = pd.DataFrame(
            [{
                "Crm Cd 1": crm_cd1,
                "Crm Cd Desc": crm_cd_desc,
                "Weapon Used Cd": weapon_used_cd,
                "Premis Cd": premis_cd,
                "Weapon Desc": weapon_desc,
                "Vict Age": vict_age,
                "Vict Descent": vict_descent,
                "Vict Sex": vict_sex,
                "Status Desc": status_desc
            }]
        )

        try:

            X_sample = encode_and_transform(
                sample
            )

            pred = model.predict(
                X_sample
            )[0]

            # Decode prediction safely.
            try:

                original_class = (
                    le_y.inverse_transform(
                        [int(pred)]
                    )[0]
                )

            except Exception:

                original_class = pred

        except Exception as e:

            st.error(
                f"Prediction failed:\n\n{e}"
            )

            st.stop()

        # ----------------------------------------------------
        # Prediction Label
        # ----------------------------------------------------

        if int(original_class) == 1:

            pred_label = "Part 1"
            explanation = (
                "The model classified this incident "
                "under the Part 1 crime category."
            )

        elif int(original_class) == 2:

            pred_label = "Part 2"
            explanation = (
                "The model classified this incident "
                "under the Part 2 crime category."
            )

        else:

            pred_label = str(original_class)
            explanation = "Predicted class from the trained model."

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        st.divider()

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Category",
                pred_label
            )

        with result_col2:

            st.metric(
                "Model",
                MODEL_DISPLAY_NAMES.get(
                    model_choice,
                    model_choice
                )
            )

        st.write(explanation)

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            try:

                probabilities = (
                    model.predict_proba(
                        X_sample
                    )[0]
                )

                prob_part1 = 0.0
                prob_part2 = 0.0

                model_classes = getattr(
                    model,
                    "classes_",
                    []
                )

                for class_value, probability in zip(
                    model_classes,
                    probabilities
                ):

                    try:
                        decoded_class = (
                            le_y.inverse_transform(
                                [int(class_value)]
                            )[0]
                        )

                    except Exception:
                        decoded_class = class_value

                    if int(decoded_class) == 1:
                        prob_part1 = float(probability)

                    elif int(decoded_class) == 2:
                        prob_part2 = float(probability)

                st.subheader("Model Confidence")

                confidence_html = f"""
                <style>
                    .confidence-wrapper {{
                        display: flex;
                        gap: 20px;
                        width: 100%;
                        margin-top: 10px;
                        margin-bottom: 20px;
                    }}

                    .confidence-card {{
                        flex: 1;
                        background: #181818;
                        border: 1px solid #333333;
                        border-radius: 12px;
                        padding: 18px;
                        box-sizing: border-box;
                    }}

                    .confidence-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 12px;
                    }}

                    .confidence-title {{
                        color: #f5f5f5;
                        font-size: 18px;
                        font-weight: 700;
                    }}

                    .confidence-value {{
                        color: #f1c40f;
                        font-size: 20px;
                        font-weight: 800;
                    }}

                    .bar-background {{
                        width: 100%;
                        height: 16px;
                        background: #2a2a2a;
                        border-radius: 10px;
                        overflow: hidden;
                    }}

                    .bar-fill {{
                        height: 100%;
                        background: linear-gradient(
                            90deg,
                            #d35400,
                            #f39c12
                        );
                        border-radius: 10px;
                    }}

                    .confidence-description {{
                        color: #bdbdbd;
                        font-size: 13px;
                        margin-top: 9px;
                    }}
                </style>

                <div class="confidence-wrapper">

                    <div class="confidence-card">

                        <div class="confidence-header">

                            <span class="confidence-title">
                                🟠 Part 1
                            </span>

                            <span class="confidence-value">
                                {prob_part1:.1%}
                            </span>

                        </div>

                        <div class="bar-background">
                            <div
                                class="bar-fill"
                                style="width: {prob_part1 * 100:.2f}%"
                            ></div>
                        </div>

                        <div class="confidence-description">
                            Major/serious crime category.
                        </div>

                    </div>


                    <div class="confidence-card">

                        <div class="confidence-header">

                            <span class="confidence-title">
                                🟠 Part 2
                            </span>

                            <span class="confidence-value">
                                {prob_part2:.1%}
                            </span>

                        </div>

                        <div class="bar-background">
                            <div
                                class="bar-fill"
                                style="width: {prob_part2 * 100:.2f}%"
                            ></div>
                        </div>

                        <div class="confidence-description">
                            Other crime category.
                        </div>

                    </div>

                </div>
                """

                components.html(
                    confidence_html,
                    height=145
                )

            except Exception as e:

                st.warning(
                    f"Could not calculate prediction probabilities: {e}"
                )
# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "Visualizations":

    st.title("Visual Analytics")

    st.write(
        f"Interactive visualizations using the deployment "
        f"sample dataset. Up to {SAMPLE_SIZE:,} rows are loaded."
    )

    tab1, tab2 = st.tabs(
        [
            "Crime Map",
            "Time Heatmap",
        ]
    )

    # --------------------------------------------------------
    # Crime Map
    # --------------------------------------------------------

    with tab1:

        st.header("Crime Map")

        map_path = Path("crime_map.html")

        if map_path.exists():

            with open(
                map_path,
                "r",
                encoding="utf-8"
            ) as f:

                html_data = f.read()

            components.html(
                html_data,
                height=600
            )

        else:

            st.warning(
                "`crime_map.html` was not found."
            )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    with tab2:

        st.header(
            "Heatmap: Day of Week × Hour of Day"
        )

        df_full = load_data()

        if df_full is None:

            st.warning(
                f"Dataset `{DATA_FILE}` was not found."
            )

        else:

            if "TIME OCC" in df_full.columns:

                def extract_hour(value):

                    if pd.isna(value):
                        return 0

                    try:
                        return int(
                            str(
                                int(value)
                            ).zfill(4)[:2]
                        )

                    except Exception:
                        return 0

                df_full["Hour OCC"] = (
                    df_full["TIME OCC"]
                    .apply(extract_hour)
                )

            else:

                df_full["Hour OCC"] = 0

            df_full["day_of_week OCC"] = (
                df_full["DATE OCC"]
                .dt.dayofweek
            )

            pivot = pd.pivot_table(
                df_full,
                values="DR_NO",
                index="day_of_week OCC",
                columns="Hour OCC",
                aggfunc="count",
                fill_value=0
            )

            pivot.index = pivot.index.map({
                0: "Mon",
                1: "Tue",
                2: "Wed",
                3: "Thu",
                4: "Fri",
                5: "Sat",
                6: "Sun"
            })

            pivot_long = (
                pivot
                .reset_index()
                .melt(
                    id_vars="day_of_week OCC",
                    var_name="Hour",
                    value_name="Crime_Count"
                )
            )

            fig = px.density_heatmap(
                pivot_long,
                x="Hour",
                y="day_of_week OCC",
                z="Crime_Count",
                color_continuous_scale="YlOrRd",
                labels={
                    "Crime_Count":
                    "Number of Crimes"
                }
            )

            fig.update_layout(
                height=500
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    st.write(
        "Compare the deployed models using accuracy, "
        "classification metrics, confusion matrices, "
        "and classification reports."
    )

    results_dict = load_results()

    if results_dict is None:

        st.warning(
            "No `results_dict.pkl` was found."
        )

        st.stop()

    ordered_models = get_ordered_models(
        results_dict
    )

    if not ordered_models:

        st.warning(
            "No results were found for the deployed models."
        )

        st.stop()

    # --------------------------------------------------------
    # Accuracy Overview
    # --------------------------------------------------------

    st.subheader("Model Accuracy")

    accuracy_rows = []

    for model_name in ordered_models:

        accuracy = get_model_accuracy(
            results_dict,
            model_name
        )

        if accuracy is not None:

            accuracy_rows.append(
                {
                    "Model":
                    MODEL_DISPLAY_NAMES.get(
                        model_name,
                        model_name
                    ),

                    "Accuracy":
                    f"{accuracy:.2%}"
                }
            )

    if accuracy_rows:

        accuracy_df = pd.DataFrame(
            accuracy_rows
        )

        st.dataframe(
            accuracy_df,
            hide_index=True,
            width="stretch"
        )

    # --------------------------------------------------------
    # Model Selection
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "Select model to inspect",
        options=ordered_models,
        index=0,
        format_func=lambda x:
        MODEL_DISPLAY_NAMES.get(x, x)
    )

    res = results_dict[selected_model]

    selected_accuracy = get_model_accuracy(
        results_dict,
        selected_model
    )

    if selected_accuracy is not None:

        st.metric(
            "Accuracy",
            f"{selected_accuracy:.2%}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.subheader("Metrics")

    metrics = res.get(
        "metrics",
        {}
    )

    if metrics:

        metrics_df = pd.DataFrame(
            [metrics]
        )

        st.dataframe(
            metrics_df,
            width="stretch"
        )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = res.get(
        "confusion_matrix"
    )

    if cm is not None:

        class_names = res.get(
            "class_names",
            ["Part 1", "Part 2"]
        )

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(
            fig,
            clear_figure=True
        )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    st.subheader("Classification Report")

    cls_report = res.get(
        "classification_report"
    )

    if cls_report is not None:

        cls_df = pd.DataFrame(
            cls_report
        ).transpose()

        st.dataframe(
            cls_df,
            width="stretch"
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title("About this App")

    st.markdown(
        """
        ## 🦇 LA Crime Explorer

        A Streamlit application built around the
        Los Angeles Police Department crime dataset.

        ### What can you do?

        - Explore crime patterns and trends.
        - Analyze crime locations and time patterns.
        - Predict the LAPD crime category: **Part 1 or Part 2**.
        - Compare multiple machine learning models.

        ### Machine Learning Models

        - XGBoost
        - Random Forest
        - Logistic Regression

        ### Prediction Target

        The model predicts the `Part 1-2` target from the
        original dataset.

        **Part 1** refers to crimes classified under the
        Part 1 category.

        **Part 2** refers to other offenses classified under
        the Part 2 category.

        These categories are **not a direct
        Violent vs Non-Violent classification**.

        ### Prediction Pipeline

        The prediction page uses the same saved preprocessing
        artifacts used during model training:

        - Label Encoders
        - Standard Scalers
        - Robust Scaler
        - Yeo-Johnson transformation

        This keeps deployment preprocessing consistent
        with the training pipeline.

        ### Dataset

        The application uses a deployment sample of the
        original LAPD crime dataset to keep the application
        lightweight.

        """
    )

    st.subheader("How to Run")

    st.code(
        """
pip install -r requirements.txt
streamlit run app.py
"""
    )

    st.write(
        "**Main Requirements:** "
        "Streamlit • Pandas • NumPy • Matplotlib • "
        "Seaborn • Plotly • Scikit-learn • XGBoost"
    )
