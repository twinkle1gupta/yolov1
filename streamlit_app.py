import streamlit as st
import pandas as pd
from PIL import Image
from src2 import run_anchor_box_pipeline  # make sure src.py is in the same folder or accessible

st.title("📦 Anchor Box Generator")

# Upload CSV file
uploaded_file = st.file_uploader("📤 Upload Bounding Box CSV (with width & height columns)", type="csv")

# Choose number of anchor boxes
k = st.slider("🔢 Number of Anchor Boxes", min_value=3, max_value=15, value=9)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.write(df.head())
    if 'width' not in df.columns or 'height' not in df.columns:
        st.error("The uploaded CSV file must contain 'width' and 'height' columns.")
    else:
        if st.button("🚀 Generate Anchor Boxes"):
            with st.spinner("Processing..."):
                result = run_anchor_box_pipeline(df, k=k)

            st.success("Anchor Boxes Generated!")

            st.subheader("📍 Anchor Boxes")
            st.write(result["clusters"])

            st.subheader("📈 Average IoU")
            st.write(f"{result['avg_iou']:.4f}")

            st.subheader("🖼️ Anchor Box Distribution")
            st.image("anchor_boxes.png", caption="Anchor Box Clustering Result", use_column_width=True)
