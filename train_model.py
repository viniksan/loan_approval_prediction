import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("loan_data.csv")

# Fill missing values
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
        df[col] = df[col].fillna(df[col].median())
    except:
        df[col] = df[col].fillna(df[col].mode()[0])

# Drop Loan_ID only if it exists
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Encode categorical columns
encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save model and encoders
joblib.dump(model, "loan_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("Model saved successfully.")