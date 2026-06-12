import pandas as pd
import streamlit as st
from PIL import Image

from utils import (
    get_available_models,
    call_prediction_api,
    draw_detections,
)

st.set_page_config(
    page_title="FaceMask Detection Demo",
    layout="wide",
)

st.title("FaceMask Detection Demo")

available_models = get_available_models()

if not available_models:
    st.error("No ONNX models found.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        original_image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            result = call_prediction_api(
                uploaded_file,
                selected_model,
            )

        detections = result.get(
            "detections",
            [],
        )

        detected_image = draw_detections(
            original_image,
            detections,
        )

        # col1, col2 = st.columns(2)

        # with col1:
        #     st.subheader("Original Image")
        #     st.image(
        #         original_image,
        #         use_container_width=True,
        #     )

        # with col2:
        #     st.subheader("Detection Result")
        #     st.image(
        #         detected_image,
        #         use_container_width=True,
        #     )

        st.subheader("Detection Result")

        st.image(
            detected_image,
            use_container_width=True
        )

        st.subheader("Prediction Summary")

        st.write(
            f"Model: {result.get('model_name')}"
        )

        st.write(
            f"Detections: {result.get('num_detections')}"
        )

        if detections:
            rows = []

            for det in detections:
                rows.append(
                    {
                        "Class": det["class_name"],
                        "Confidence": round(
                            det["confidence"],
                            4,
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
            )

        else:
            st.info("No objects detected.")