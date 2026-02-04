import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv("data/churn.csv")

# Select ONLY required columns
df = df[[
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "Churn"
]]

# Handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode categorical features
le_contract = LabelEncoder()
le_payment = LabelEncoder()
le_churn = LabelEncoder()

df["Contract"] = le_contract.fit_transform(df["Contract"])
df["PaymentMethod"] = le_payment.fit_transform(df["PaymentMethod"])
df["Churn"] = le_churn.fit_transform(df["Churn"])

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Save everything
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model retrained & saved successfully")
