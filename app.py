import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Set page config for a premium look
st.set_page_config(
    page_title="Sri Lanka University Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .z-score-val {
        font-size: 3rem;
        font-weight: bold;
        color: #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('models/model.pkl')
    encoder = joblib.load('models/encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    metadata = joblib.load('models/metadata.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, encoder, scaler, metadata, feature_names

def main():
    st.title("🎓 Sri Lankan UGC University Cutoff Predictor")
    st.markdown("Estimate the required **Z-Score** for your desired university and course based on historical trends.")

    if not os.path.exists('models/model.pkl'):
        st.error("Model files not found. Please run `data_prep.py` and `train.py` first.")
        return

    model, encoder, scaler, metadata, feature_names = load_assets()

    # Sidebar Inputs
    st.sidebar.header("Student details & Choices")
    
    district = st.sidebar.selectbox("Select District", metadata['districts'])
    stream = st.sidebar.selectbox("Select A/L Stream", metadata['streams'])
    course = st.sidebar.selectbox("Desired Course", metadata['courses'])
    university = st.sidebar.selectbox("Desired University", metadata['universities'])
    year = st.sidebar.selectbox("Exam Year", sorted(metadata['years'], reverse=True))
    intake = st.sidebar.selectbox("Intake", sorted(metadata['intakes'], reverse=True))

    # Prepare input data
    input_df = pd.DataFrame([{
        'Exam Year': year,
        'District': district,
        'University': university,
        'Course': course,
        'Stream': stream,
        'Intake': intake
    }])

    # Prediction
    if st.sidebar.button("Predict Cutoff Z-Score"):
        # 1. Encoding
        # The encoder expects categorical columns to be transformed. 
        # TargetEncoder needs to handle the input dataframe
        input_encoded = encoder.transform(input_df)
        
        # 2. Scaling
        input_scaled = scaler.transform(input_encoded)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        # 3. Model Prediction
        prediction = model.predict(input_scaled_df)[0]

        # Display Result
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Z-Score Cutoff</h3>
                    <div class="z-score-val">{prediction:.4f}</div>
                    <p>For {course} at {university}</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.info("💡 Note: This is an AI-driven estimate based on historical UGC data. Actual cutoffs may vary due to various socio-economic factors.")

        # SHAP Explanation Section
        st.markdown("### 🔍 Why this prediction?")
        st.write("The following plot shows how each factor contributed to the predicted Z-score. Factors in red pushed the score higher, while those in blue pushed it lower.")

        # Create SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled_df)

        # Waterfall plot for single prediction
        fig, ax = plt.subplots(figsize=(10, 4))
        # shap.waterfall_plot expects an Explanation object or base_values + shap_values
        # For Trees, we can use force_plot or waterfall
        # waterfall_plot needs shap.Explanation
        exp = shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value[0], 
            data=input_scaled_df.iloc[0], 
            feature_names=feature_names
        )
        shap.waterfall_plot(exp, show=False)
        st.pyplot(plt.gcf())
        plt.close()

    else:
        st.info("Select your details in the sidebar and click **Predict Cutoff Z-Score** to see the results.")

    # Additional visualizations from training
    with st.expander("Model Global Insights (SHAP Summary)"):
        st.markdown("These plots show the overall influence of features across the entire dataset.")
        col_im1, col_im2 = st.columns(2)
        if os.path.exists('models/shap_importance_bar.png'):
            col_im1.image('models/shap_importance_bar.png', caption="Feature Importance")
        if os.path.exists('models/shap_summary_beeswarm.png'):
            col_im2.image('models/shap_summary_beeswarm.png', caption="Impact on Prediction (Beeswarm)")

if __name__ == "__main__":
    main()
