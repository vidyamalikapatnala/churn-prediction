import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.title("Customer Churn Prediction")
st.write("Customer velthada leda ani predict cheyandi")

df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df[["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]]
y = df["Churn"]

categorical = ["Contract", "InternetService"]
numeric = ["tenure", "MonthlyCharges", "TotalCharges"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

model.fit(X, y)

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict"):
    input_data = pd.DataFrame([[tenure, monthly, total, contract, internet]],
        columns=["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"])
    prediction = model.predict(input_data)[0]
    if prediction == "Yes":
        st.error("Customer Churn Avvachu!")
    else:
        st.success("Customer Stay Avthadu!")
