import streamlit as st
import pandas as pd
import io
from utils.classification_utils import (
    train_classification_model,
    save_model,
    load_model
)

st.title("Create Classification Model")
st.write("Create a classification model for automatic ticket classification")

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'df' not in st.session_state:
    st.session_state.df = None

data_preprocessing_tab, model_training_tab, model_eval_tab, save_model_tab = st.tabs(["Data Preprocessing", "Model Training", "Model Evaluation", "Save Model"])

with data_preprocessing_tab:
    st.subheader("Data Preprocessing")
    st.write("Upload a CSV file with 'text' and 'category' columns for training")
    st.info("Expected format: CSV with columns 'text' (ticket content) and 'category' (HR Support, IT Support, Transportation Support)")

    upload_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
    
    if upload_file is not None:
        try:
            # Convert uploaded file to string
            stringio = io.StringIO(upload_file.getvalue().decode("utf-8"))
            st.session_state.df = pd.read_csv(stringio, header=None, names=['text', 'category'])
            
            st.success("Data loaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head())
            
            st.subheader("Data Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(st.session_state.df))
            with col2:
                st.metric("Categories", st.session_state.df['category'].nunique())
            
            st.subheader("Category Distribution")
            category_counts = st.session_state.df['category'].value_counts()
            st.bar_chart(category_counts)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.error("Please ensure your CSV has 'text' and 'category' columns.")

with model_training_tab:
    st.subheader("Model Training")
    st.write("Train the classification model using TF-IDF and SVM")
    
    train_model_button = st.button("Train Model", key="train_btn")
    
    if train_model_button:
        if st.session_state.df is not None:
            try:
                with st.spinner("Training model..."):
                    st.session_state.trained_model = train_classification_model(st.session_state.df)
                st.success("Model trained successfully!")
                st.metric("Training Accuracy", f"{st.session_state.trained_model['accuracy']:.3f}")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.error("Please try again.")
        else:
            st.error("Please upload and load data first.")

with model_eval_tab:
    st.subheader("Model Evaluation")
    st.write("Evaluate the model performance")
    
    if st.session_state.trained_model is not None:
        st.subheader("Model Performance")
        
        accuracy = st.session_state.trained_model['accuracy']
        st.metric("Accuracy", f"{accuracy:.3f}")
        
        st.subheader("Classification Report")
        report = st.session_state.trained_model['classification_report']
        
        # Create a DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Show test predictions
        st.subheader("Sample Predictions")
        X_test, y_test, y_pred = st.session_state.trained_model['test_data']
        
        sample_df = pd.DataFrame({
            'Text': X_test.head(10),
            'Actual': y_test.head(10),
            'Predicted': y_pred[:10]
        })
        st.dataframe(sample_df)
        
    else:
        st.info("Train a model first to see evaluation results.")

with save_model_tab:
    st.subheader("Save Model")
    st.write("Save the trained model for use in classification")
    
    save_model_button = st.button("Save Model", key="save_btn")
    
    if save_model_button:
        if st.session_state.trained_model is not None:
            try:
                with st.spinner("Saving model..."):
                    save_model(st.session_state.trained_model)
                st.success("Model saved successfully!")
                st.info("Model can now be used for automatic ticket classification.")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
                st.error("Please try again.")
        else:
            st.error("Please train a model first.")
    
    # Show if model exists
    existing_model = load_model()
    if existing_model is not None:
        st.info("✅ A trained model is available for classification.")
    else:
        st.warning("❌ No trained model found. Train and save a model to enable classification.")
