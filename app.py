
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Credit Default Predictor",
    layout="wide"
)

st.title("Credit Card Default Classification")
st.write(
    "Compare five machine learning classifiers "
    "on the uploaded test dataset."
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# Upload test dataset
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Dataset shape:", data.shape)
    st.dataframe(data.head())

    target_column = "default payment next month"

    if target_column not in data.columns:
        st.error(f"Missing target column: {target_column}")
    else:
        X_test = data.drop(columns=[target_column])
        y_test = data[target_column]

        selected_model = st.selectbox(
            "Select Classification Model",
            list(MODEL_FILES.keys())
        )

        model_path = os.path.join(
            "model",
            MODEL_FILES[selected_model]
        )

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_score),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }
            st.subheader("Model Evaluation Metrics")
            columns = st.columns(3)
            for i, (metric, value) in enumerate(metrics.items()):
                columns[i % 3].metric(metric, f"{value:.4f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: No Default", "Actual: Default"],
                columns=["Predicted: No Default", "Predicted: Default"]
            )
            st.dataframe(cm_df)

            st.subheader("Classification Report")
            report = classification_report(
                y_test,
                y_pred,
                target_names=["No Default", "Default"],
                output_dict=True,
                zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose())

        else:
            st.error("Selected model file was not found.")

