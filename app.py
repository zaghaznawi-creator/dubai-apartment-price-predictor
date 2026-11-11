import streamlit as st
import pandas as pd
import joblib

# Load the saved models and preprocessing information
@st.cache_resource
def load_models():
    return joblib.load("dubai_price_models.pkl")

bundle = load_models()

st.set_page_config(
    page_title="Dubai Apartment Price Predictor",
    page_icon="🏙️"
)

st.title("🏙️ Dubai Apartment Price Predictor")

st.write(
    "Estimate the selling price of an apartment in Dubai "
    "using a machine learning regression model."
)

# Apartment details
st.subheader("Enter Apartment Details")

location = st.selectbox(
    "Apartment Location",
    sorted(bundle["common_areas"])
)

size = st.number_input(
    "Apartment Size (m²)",
    min_value=10.0,
    max_value=1000.0,
    value=100.0,
    step=5.0
)

bedrooms = st.selectbox(
    "Number of Bedrooms",
    [0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: "Studio" if x == 0 else str(x)
)

month = st.selectbox(
    "Transaction Month",
    list(range(1, 13)),
    format_func=lambda x: [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ][x - 1]
)

model_choice = st.radio(
    "Prediction Model",
    ["Non-Luxury Model", "Original Model"]
)

# Predict apartment price
if st.button("Predict Apartment Price", type="primary"):

    input_area = pd.DataFrame({
        "AREA_GROUPED": [location]
    })

    encoded_area = bundle["encoder"].transform(input_area)

    area_df = pd.DataFrame(
        encoded_area,
        columns=bundle["encoder"].get_feature_names_out()
    )

    numeric_df = pd.DataFrame({
        "ACTUAL_AREA": [size],
        "BEDROOMS": [bedrooms],
        "MONTH": [month]
    })

    input_features = pd.concat(
        [numeric_df, area_df],
        axis=1
    )

    input_features = input_features.reindex(
        columns=bundle["feature_columns"],
        fill_value=0
    )

    if model_choice == "Non-Luxury Model":
        model = bundle["non_luxury_model"]
    else:
        model = bundle["original_model"]

    predicted_price = model.predict(input_features)[0]

    if predicted_price > 0:
        st.success("Estimated Apartment Selling Price")
        st.metric(
            "Predicted Price",
            f"AED {predicted_price:,.0f}"
        )
    else:
        st.warning(
            "The model could not produce a meaningful "
            "positive price for these inputs."
        )

st.caption(
    "Predictions are estimates based on historical 2026 "
    "Dubai apartment transactions, not official valuations. "
    "Accuracy may vary by apartment and location."
)
