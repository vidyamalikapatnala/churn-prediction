import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_csv("churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Try a new customer
sample = pd.DataFrame([{
    "tenure": 5,
    "MonthlyCharges": 80.0,
    "TotalCharges": 400.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic"
}])
print("Predicted churn:", model.predict(sample)[0])
