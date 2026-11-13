
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path


# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="India Disaster Prediction",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.title("🌍 India Environmental Disaster Prediction")
st.caption("Machine Learning Based Environmental Disaster Analysis")


# 2. LOAD DATA AND MODEL


@st.cache_data
def load_data():
    df = pd.read_excel(
    BASE_DIR / "India_Environmental_Disaster_Analytics_All_Data_2015_2025_5500.xlsx",
    engine="openpyxl"
)
    

    X = pd.read_csv(
        BASE_DIR / "Disaster_ML_Features.csv"
    )

    results_path = BASE_DIR / "ML_Model_Comparison.csv"

    if results_path.exists():
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame()

    return df, X, results


@st.cache_resource
def load_model():
    model = joblib.load(
        BASE_DIR / "Best_Disaster_Prediction_Model.pkl"
    )

    encoder_path = BASE_DIR / "Risk_Label_Encoder.pkl"

    encoder = None

    if encoder_path.exists():
        encoder = joblib.load(encoder_path)

    return model, encoder


try:
    df, X, results = load_data()
    model, le = load_model()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# Clean column names
df.columns = df.columns.str.strip()
X.columns = X.columns.str.strip()

# Remove accidental CSV index columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
X = X.loc[:, ~X.columns.str.contains("^Unnamed")]


# 3. SIDEBAR NAVIGATION


st.sidebar.title("🌍 Navigation")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Home",
        "Risk Prediction",
        "Disaster Map",
        "Model Performance",
        "Dataset Analysis",
        "Feature Importance"
    ]
)


# 4. HOME PAGE


if page == "Home":

    st.header("Welcome to Disaster Analytics")

    st.write(
        "Explore environmental disaster records "
        "and risk classifications across Indian states."
    )

    # State selection
    if "State" in df.columns:

        states = sorted(
            df["State"].dropna().astype(str).unique()
        )

        selected_state = st.selectbox(
            "Select Your State",
            ["All States"] + states
        )

        if selected_state == "All States":
            filtered_df = df.copy()
        else:
            filtered_df = df[
                df["State"].astype(str) == selected_state
            ].copy()

    else:
        filtered_df = df.copy()
        st.warning("State column is not available.")

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(filtered_df):,}"
        )

    with col2:
        if "State" in filtered_df.columns:
            st.metric(
                "Total States",
                filtered_df["State"].nunique()
            )

    with col3:
        if "DisasterType" in filtered_df.columns:
            st.metric(
                "Disaster Types",
                filtered_df["DisasterType"].nunique()
            )

    with col4:
        if "RiskLevel" in filtered_df.columns:
            st.metric(
                "Risk Levels",
                filtered_df["RiskLevel"].nunique()
            )

    st.divider()

    # Map
    st.subheader("🗺️ State-wise Disaster Map")

    if {"Latitude", "Longitude"}.issubset(filtered_df.columns):

        map_df = filtered_df.copy()

        map_df["Latitude"] = pd.to_numeric(
            map_df["Latitude"], errors="coerce"
        )

        map_df["Longitude"] = pd.to_numeric(
            map_df["Longitude"], errors="coerce"
        )

        map_df = map_df.dropna(
            subset=["Latitude", "Longitude"]
        )

        map_df = map_df[
            map_df["Latitude"].between(6, 38) &
            map_df["Longitude"].between(68, 98)
        ]

        if not map_df.empty:

            color_column = None

            if "RiskLevel" in map_df.columns:
                color_column = "RiskLevel"

            elif "DisasterType" in map_df.columns:
                color_column = "DisasterType"

            fig = px.scatter_map(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color=color_column,
                hover_data=[
                    c for c in [
                        "State",
                        "District",
                        "DisasterType",
                        "RiskLevel"
                    ]
                    if c in map_df.columns
                ],
                zoom=3.5,
                height=550,
                map_style="open-street-map",
                title="Environmental Disaster Locations"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning("No valid coordinates found.")

    else:
        st.warning("Latitude and Longitude columns are missing.")

    # State records
    st.subheader("📋 State Disaster Records")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# 5. RISK PREDICTION PAGE


elif page == "Risk Prediction":

    st.header("🤖 Machine Learning Risk Prediction")

    st.write(
        "Select a processed dataset record to demonstrate "
        "the trained model's risk classification."
    )

    # Model feature columns
    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)
    else:
        expected_columns = list(X.columns)

    missing_columns = [
        col for col in expected_columns
        if col not in X.columns
    ]

    if missing_columns:

        st.error(
            f"Features missing from Disaster_ML_Features.csv: "
            f"{missing_columns}"
        )

    else:

        # Match exact training columns
        prediction_data = X.reindex(
            columns=expected_columns
        )

        record_number = st.number_input(
            "Select Record Number",
            min_value=0,
            max_value=len(prediction_data) - 1,
            value=0,
            step=1
        )

        record_number = int(record_number)

        st.subheader("Selected Record")

        st.dataframe(
            prediction_data.iloc[[record_number]],
            use_container_width=True
        )

        if st.button("Predict Disaster Risk"):

            try:
                input_data = prediction_data.iloc[
                    [record_number]
                ]

                prediction = model.predict(input_data)

                predicted_value = prediction[0]

                if le is not None:
                    predicted_risk = le.inverse_transform(
                        prediction
                    )[0]
                else:
                    predicted_risk = predicted_value

                st.success(
                    f"Predicted Disaster Risk: {predicted_risk}"
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")



# 6. DISASTER MAP PAGE


elif page == "Disaster Map":

    st.header("🗺️ India Disaster Map")

    if "State" in df.columns:

        states = sorted(
            df["State"].dropna().astype(str).unique()
        )

        selected_state = st.selectbox(
            "Choose State for Map",
            ["All States"] + states,
            key="map_state"
        )

        if selected_state == "All States":
            map_df = df.copy()
        else:
            map_df = df[
                df["State"].astype(str) == selected_state
            ].copy()

        if {"Latitude", "Longitude"}.issubset(map_df.columns):

            map_df["Latitude"] = pd.to_numeric(
                map_df["Latitude"], errors="coerce"
            )

            map_df["Longitude"] = pd.to_numeric(
                map_df["Longitude"], errors="coerce"
            )

            map_df = map_df.dropna(
                subset=["Latitude", "Longitude"]
            )

            map_df = map_df[
                map_df["Latitude"].between(6, 38) &
                map_df["Longitude"].between(68, 98)
            ]

            if not map_df.empty:

                color_column = (
                    "RiskLevel" if "RiskLevel" in map_df.columns
                    else "DisasterType" if "DisasterType" in map_df.columns
                    else None
                )

                fig = px.scatter_map(
                    map_df,
                    lat="Latitude",
                    lon="Longitude",
                    color=color_column,
                    hover_data=[
                        c for c in [
                            "State",
                            "District",
                            "DisasterType",
                            "RiskLevel"
                        ]
                        if c in map_df.columns
                    ],
                    zoom=5,
                    height=600,
                    map_style="open-street-map"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.warning("No map locations available.")

        else:
            st.error("Coordinate columns are missing.")



# 7. MODEL PERFORMANCE PAGE


elif page == "Model Performance":

    st.header("📊 Model Performance Comparison")

    if not results.empty:

        st.dataframe(
            results,
            use_container_width=True
        )

        if {"Model", "Accuracy"}.issubset(results.columns):

            fig = px.bar(
                results,
                x="Model",
                y="Accuracy",
                color="Model",
                title="Machine Learning Model Accuracy",
                text_auto=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            best_row = results.loc[
                results["Accuracy"].idxmax()
            ]

            st.success(
                f"Best Model: {best_row['Model']} "
                f"| Accuracy: {best_row['Accuracy']}"
            )

    else:
        st.warning("Model comparison file is not available.")



# 8. DATASET ANALYSIS PAGE

elif page == "Dataset Analysis":

    st.header("📈 Environmental Dataset Analysis")

    st.write("Dataset Shape:", df.shape)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    # Disaster type chart
    if "DisasterType" in df.columns:

        st.subheader("Disaster Type Distribution")

        disaster_counts = (
            df["DisasterType"]
            .value_counts()
            .reset_index()
        )

        disaster_counts.columns = [
            "Disaster Type",
            "Count"
        ]

        fig = px.pie(
            disaster_counts,
            names="Disaster Type",
            values="Count",
            title="Disaster Type Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Risk level chart
    if "RiskLevel" in df.columns:

        st.subheader("Risk Level Distribution")

        fig = px.histogram(
            df,
            x="RiskLevel",
            color="RiskLevel",
            title="Disaster Risk Levels"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Rainfall chart
    if "Rainfall_mm" in df.columns:

        st.subheader("Rainfall Analysis")

        fig = px.histogram(
            df,
            x="Rainfall_mm",
            title="Rainfall Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



# 9. FEATURE IMPORTANCE PAGE


elif page == "Feature Importance":

    st.header("⭐ Feature Importance Analysis")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        fig = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Feature Importance Details")

        st.dataframe(
            importance_df,
            use_container_width=True
        )

    else:
        st.warning(
            "Feature importance is not supported by this model."
        )



# 10. FOOTER


st.divider()

st.caption(
    "India Environmental Disaster Prediction | "
    "Academic Machine Learning Project"
)

st.caption(
    "Note: This project uses synthetic sample data. "
    "Map coordinates and predictions are for educational "
    "demonstration only, not real disaster warnings."
)
