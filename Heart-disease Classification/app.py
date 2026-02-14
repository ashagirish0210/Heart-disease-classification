
import sys

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Title
st.title("Heart Disease Prediction System")
st.markdown("---")

# Load models
@st.cache_resource
def load_model(model_name):
    model_path = f'models_pkl/{model_name}_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

# Preprocess uploaded data
def preprocess_uploaded_data(df, label_encoders):
    df = df.drop(['id', 'dataset'], axis=1, errors='ignore')

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].map(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

    df = df.fillna(df.median(numeric_only=True))

    return df

# Sidebar
st.sidebar.header("Model Selection")
model_options = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest (Ensemble)": "random_forest",
    "XGBoost (Ensemble)": "xgboost"
}

selected_model_name = st.sidebar.selectbox("Select a Model", list(model_options.keys()))
selected_model_key = model_options[selected_model_name]

# Load selected model
model_data = load_model(selected_model_key)

if model_data:
    st.sidebar.success(f"{selected_model_name} loaded successfully!")

    # Display model metrics
    st.sidebar.markdown(" Model Performance Metrics")
    metrics = model_data['metrics']
    st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    st.sidebar.metric("AUC Score", f"{metrics['auc']:.4f}")
    st.sidebar.metric("Precision", f"{metrics['precision']:.4f}")
    st.sidebar.metric("Recall", f"{metrics['recall']:.4f}")
    st.sidebar.metric("F1 Score", f"{metrics['f1']:.4f}")
    st.sidebar.metric("MCC Score", f"{metrics['mcc']:.4f}")
else:
    st.sidebar.error(f"{selected_model_name} not found. Please train the model first.")

# Main content
st.markdown(" Data Upload and Prediction")

# Create sample test data
st.markdown(" Download Test Data Template")
if st.button("Download Sample Test Data"):
    sample_data = pd.DataFrame({
        'age': [63, 67, 45],
        'sex': ['Male', 'Female', 'Male'],
        'cp': ['typical angina', 'asymptomatic', 'non-anginal'],
        'trestbps': [145, 160, 120],
        'chol': [233, 286, 220],
        'fbs': ['TRUE', 'FALSE', 'FALSE'],
        'restecg': ['lv hypertrophy', 'normal', 'normal'],
        'thalch': [150, 108, 170],
        'exang': ['FALSE', 'TRUE', 'FALSE'],
        'oldpeak': [2.3, 1.5, 1.0],
        'slope': ['downsloping', 'flat', 'upsloping'],
        'ca': [0, 3, 1],
        'thal': ['fixed defect', 'normal', 'reversable defect'],
        'num': [0, 2, 1]
    })

    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="test_data_sample.csv",
        mime="text/csv"
    )

st.markdown("---")

# File upload
st.markdown("Upload Your Test Data")
uploaded_file = st.file_uploader("Upload CSV file with test data", type=['csv'])

if uploaded_file and model_data:
    try:
        # Read uploaded file
        test_df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully! Shape: {test_df.shape}")

        # Display uploaded data
        st.markdown("#### Uploaded Data Preview")
        st.dataframe(test_df.head(10))

        # Check if target column exists
        has_target = 'num' in test_df.columns

        if has_target:
            y_true = (test_df['num'] > 0).astype(int)
            X_test = test_df.drop('num', axis=1)
        else:
            X_test = test_df.copy()

        # Preprocess data
        X_test_processed = preprocess_uploaded_data(X_test.copy(), model_data['label_encoders'])

        # Ensure columns match
        expected_features = model_data['feature_names']
        for col in expected_features:
            if col not in X_test_processed.columns:
                X_test_processed[col] = 0

        X_test_processed = X_test_processed[expected_features]

        # Make predictions
        pipeline = model_data['pipeline']
        predictions = pipeline.predict(X_test_processed)
        predictions_proba = pipeline.predict_proba(X_test_processed)

        # Display predictions
        st.markdown("---")
        st.markdown("Prediction Results")

        results_df = X_test.copy()
        results_df['Predicted_Class'] = predictions
        results_df['Probability_No_Disease'] = predictions_proba[:, 0]
        results_df['Probability_Disease'] = predictions_proba[:, 1]
        results_df['Prediction'] = results_df['Predicted_Class'].map({0: 'No Disease', 1: 'Disease'})

        st.dataframe(results_df)

        # Download predictions
        csv_results = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv_results,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # If target exists, show evaluation
        if has_target:
            st.markdown("---")
            st.markdown("Model Evaluation on Uploaded Data")

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_true, predictions):.4f}")
                st.metric("Precision", f"{precision_score(y_true, predictions, average='weighted'):.4f}")

            with col2:
                st.metric("Recall", f"{recall_score(y_true, predictions, average='weighted'):.4f}")
                st.metric("F1 Score", f"{f1_score(y_true, predictions, average='weighted'):.4f}")

            with col3:
                st.metric("MCC Score", f"{matthews_corrcoef(y_true, predictions):.4f}")
                st.metric("AUC Score", f"{roc_auc_score(y_true, predictions_proba[:, 1]):.4f}")

            # Confusion Matrix
            st.markdown("Confusion Matrix")
            cm = confusion_matrix(y_true, predictions)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'], ax=ax)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title(f'Confusion Matrix - {selected_model_name}')
            st.pyplot(fig)

            # Classification Report
            st.markdown("Classification Report")
            report = classification_report(y_true, predictions, target_names=['No Disease', 'Disease'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format and column names.")

# Information section
st.markdown("---")
st.markdown("About This Application")
st.info("""
This application implements 6 machine learning classification models for heart disease prediction:
1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree** - Tree-based model with interpretable rules
3. **K-Nearest Neighbors** - Instance-based learning algorithm
4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest** - Ensemble method using multiple decision trees
6. **XGBoost** - Gradient boosting ensemble technique

**Dataset**: UCI Heart Disease Dataset (Cleveland)
- 920 instances with 13 features
- Binary classification: Disease (1) vs No Disease (0)
""")

