import joblib
import pandas as pd

model = joblib.load('loan_model.pkl')
meta = joblib.load('label_encoders.pkl')
encoders = meta.get('encoders', {})
feature_columns = meta.get('feature_columns')

sample = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 0,
    'LoanAmount': 100,
    'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

row = {}
for col in feature_columns:
    val = sample.get(col, 0)
    if col in encoders:
        le = encoders[col]
        try:
            row[col] = le.transform([str(val)])[0]
        except Exception:
            row[col] = 0
    else:
        row[col] = val

df = pd.DataFrame([row], columns=feature_columns)
print('Input DF:')
print(df)
print('Predict Proba:', model.predict_proba(df))
print('Predict:', model.predict(df))
