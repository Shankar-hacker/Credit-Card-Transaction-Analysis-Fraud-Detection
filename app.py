
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Transaction Analysis & Fraud Detection")

# File upload
uploaded_file = st.file_uploader("📂 Upload your credit card CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.write(data.head())

    st.subheader("📉 Missing Values")
    st.write(data.isnull().sum())

    # Transaction type breakdown
    if "type" in data.columns:
        st.subheader("📊 Transaction Type Distribution")
        type_counts = data["type"].value_counts()
        summary_df = pd.DataFrame({
            "Transaction Type": type_counts.index,
            "Count": type_counts.values
        })
        fig = px.pie(
            summary_df,
            values="Count",
            names="Transaction Type",
            hole=0.5,
            title="Distribution of Transaction Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'type' column not found in dataset.")

    # Correlation analysis
    st.subheader("🔗 Feature Correlation with 'isFraud'")
    numeric_data = data.select_dtypes(include=[np.number])
    if "isFraud" in numeric_data.columns:
        correlation = numeric_data.corr()
        st.write(correlation["isFraud"].sort_values(ascending=False))
    else:
        st.warning("⚠️ 'isFraud' column not found in numeric data.")

    # Encoding
    st.subheader("🔁 Data Preprocessing")
    data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                     "CASH_IN": 3, "TRANSFER": 4,
                                     "DEBIT": 5})
    data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
    st.write(data.head())

    # Model training
    st.subheader("🤖 Train Decision Tree Model")

    try:
        x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
        y = np.array(data[["isFraud"]])

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(xtrain, ytrain)
        score = model.score(xtest, ytest)

        st.success(f"✅ Model Accuracy: {score:.2f}")

        # Prediction
        st.subheader("🔮 Predict a Transaction")
        st.markdown("Provide inputs to simulate and predict fraud detection.")

        default_input = [4, 9000.60, 9000.60, 0.0]
        input_type = st.number_input("Transaction Type (1-5)", min_value=1, max_value=5, value=default_input[0])
        input_amount = st.number_input("Amount", value=default_input[1])
        input_oldbalanceOrg = st.number_input("Old Balance (Origin)", value=default_input[2])
        input_newbalanceOrig = st.number_input("New Balance (Origin)", value=default_input[3])

        if st.button("Predict"):
            features = np.array([[input_type, input_amount, input_oldbalanceOrg, input_newbalanceOrig]])
            prediction = model.predict(features)
            st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
