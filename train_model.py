import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib


def load_data(path='dataset/loan_data.csv'):
    return pd.read_csv(path)


def preprocess(df, categorical_cols, numeric_cols):
    # Fill numeric missing with median
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing with mode
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def train(save_model_path='loan_model.pkl', save_encoders_path='label_encoders.pkl'):
    df = load_data()

    # Typical dataset columns - adjust if your dataset differs
    target = 'Loan_Status'
    categorical_cols = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'
    ]
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    df, encoders = preprocess(df, categorical_cols, numeric_cols)

    # Map target to binary
    if target in df.columns:
        le_target = LabelEncoder()
        df[target] = le_target.fit_transform(df[target].astype(str))
        encoders[target] = le_target
    else:
        raise ValueError(f"Target column '{target}' not found in dataset")

    feature_cols = [c for c in categorical_cols + numeric_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, save_model_path)
    # Save encoders together with the feature column ordering used for training
    metadata = {
        'encoders': encoders,
        'feature_columns': feature_cols
    }
    joblib.dump(metadata, save_encoders_path)

    print(f"Saved model to {save_model_path} and metadata to {save_encoders_path}")


if __name__ == '__main__':
    train()
