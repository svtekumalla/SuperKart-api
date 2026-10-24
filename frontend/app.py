import requests
import pandas as pd
import streamlit as st

# Base URL of the Flask backend
BACKEND_URL = "http://localhost:7860"

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]

CATEGORICAL_FEATURES = [
    "Product_Sugar_Content",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
]

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛒 SuperKart Sales Prediction")

st.write(
    """
    Predict total product sales using the trained SuperKart
    machine learning model. Enter product and store details
    for a single prediction, or upload a CSV file for batch predictions.
    """
)

st.divider()

# --------------------------------------------------
# Online Prediction
# --------------------------------------------------

st.subheader("Online Prediction")

with st.form("sales_prediction_form"):

    st.markdown("### Product Information")

    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=12.0,
            step=0.1,
        )

        product_sugar_content = st.text_input(
            "Product Sugar Content",
            placeholder="Enter training category",
        )

        product_allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            value=0.05,
            step=0.01,
            format="%.4f",
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

    with col2:
        product_id_char = st.text_input(
            "Product ID Character",
            placeholder="Enter training category",
        )

        product_type_category = st.text_input(
            "Product Type Category",
            placeholder="Enter training category",
        )

    st.markdown("### Store Information")

    col3, col4 = st.columns(2)

    with col3:
        store_size = st.text_input(
            "Store Size",
            placeholder="Enter training category",
        )

        store_location_city_type = st.text_input(
            "Store Location City Type",
            placeholder="Enter training category",
        )

    with col4:
        store_type = st.text_input(
            "Store Type",
            placeholder="Enter training category",
        )

        store_age_years = st.number_input(
            "Store Age (Years)",
            min_value=0,
            value=10,
            step=1,
        )

    st.caption(
        "Enter categorical values exactly as represented in your "
        "training data."
    )

    predict_clicked = st.form_submit_button(
        "Predict Sales",
        type="primary",
        use_container_width=True,
    )

if predict_clicked:

    input_data = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content.strip(),
        "Product_Allocated_Area": product_allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size.strip(),
        "Store_Location_City_Type": store_location_city_type.strip(),
        "Store_Type": store_type.strip(),
        "Product_Id_char": product_id_char.strip(),
        "Store_Age_Years": store_age_years,
        "Product_Type_Category": product_type_category.strip(),
    }

    empty_fields = [
        feature
        for feature in CATEGORICAL_FEATURES
        if not input_data[feature]
    ]

    if empty_fields:
        st.warning(
            "Please complete the following fields: "
            + ", ".join(empty_fields)
        )
    else:
        try:
            with st.spinner("Predicting sales..."):
                response = requests.post(
                    f"{BACKEND_URL}/v1/predict",
                    json=input_data,
                    timeout=30,
                )

            if response.status_code == 200:
                result = response.json()

                # This key must match the Flask API response.
                prediction = result.get(
                    "predicted_Product_Store_Sales_Total"
                )

                if prediction is None:
                    st.error(
                        "The API response did not contain the expected "
                        "sales prediction field."
                    )
                    st.json(result)
                else:
                    st.success("Sales prediction completed!")

                    st.metric(
                        label="Predicted Product Store Sales Total",
                        value=f"{float(prediction):,.2f}",
                    )

            else:
                st.error(
                    f"Prediction failed ({response.status_code})."
                )
                st.code(response.text)

        except requests.exceptions.RequestException as exc:
            st.error(
                "Unable to connect to the prediction API. "
                "Check that the Flask backend is running and "
                "BACKEND_URL is correct."
            )
            st.caption(str(exc))

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

st.divider()
st.subheader("Batch Prediction")

st.write(
    "Upload a CSV file containing the required 10 input features. "
    "The backend will generate a prediction for each row."
)

uploaded_file = st.file_uploader(
    "Upload CSV file for batch prediction",
    type=["csv"],
)

if uploaded_file is not None:

    try:
        preview_df = pd.read_csv(uploaded_file)

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in preview_df.columns
        ]

        if missing_features:
            st.error(
                "Your CSV is missing these required columns: "
                + ", ".join(missing_features)
            )
        elif preview_df.empty:
            st.warning("The uploaded CSV contains no rows.")
        else:
            st.write(f"Rows to predict: {len(preview_df)}")
            st.dataframe(
                preview_df.head(),
                use_container_width=True,
            )

            if st.button(
                "Predict Batch",
                type="primary",
            ):
                # Rewind the uploaded file before sending it.
                uploaded_file.seek(0)

                try:
                    with st.spinner("Generating batch predictions..."):
                        response = requests.post(
                            f"{BACKEND_URL}/v1/predictbatch",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "text/csv",
                                )
                            },
                            timeout=120,
                        )

                    if response.status_code == 200:
                        result = response.json()

                        st.success("Batch predictions completed!")

                        # Supports the Flask response format:
                        # {"predictions": [...]}
                        predictions = result.get("predictions", result)

                        if isinstance(predictions, list):
                            results_df = pd.DataFrame(predictions)
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                            )

                            st.download_button(
                                label="Download Predictions CSV",
                                data=results_df.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="superkart_predictions.csv",
                                mime="text/csv",
                            )
                        else:
                            st.json(predictions)

                    else:
                        st.error(
                            f"Batch prediction failed "
                            f"({response.status_code})."
                        )
                        st.code(response.text)

                except requests.exceptions.RequestException as exc:
                    st.error(
                        "Unable to connect to the prediction API. "
                        "Check the Flask backend and its URL."
                    )
                    st.caption(str(exc))

    except Exception as exc:
        st.error("Unable to read the uploaded CSV file.")
        st.caption(str(exc))

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption("SuperKart | Machine Learning Sales Prediction")
