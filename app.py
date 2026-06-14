import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(page_title='Loan Approval Prediction')

st.title('Loan Approval Prediction')

MODEL_PATH = 'loan_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'

def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(ENCODERS_PATH)
        # metadata should contain 'encoders' and 'feature_columns'
        encoders = metadata.get('encoders', {}) if isinstance(metadata, dict) else metadata
        feature_columns = metadata.get('feature_columns', None) if isinstance(metadata, dict) else None
        return model, encoders, feature_columns
    except Exception as e:
        st.error(f'Failed to load model or encoders: {e}')
        return None, None, None


model, encoders, feature_columns = load_artifacts()

with st.form('input_form'):
    st.header('Applicant information')
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Married', ['No', 'Yes'])
    Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed', ['No', 'Yes'])
    ApplicantIncome = st.number_input('Applicant Income', min_value=0)
    CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
    LoanAmount = st.number_input('Loan Amount', min_value=0.0, format="%.2f")
    LoanAmount_unit = st.selectbox('Loan Amount Unit', ['Thousands', 'Rupees', 'Lakhs', 'Crores'])
    Loan_Amount_Term = st.selectbox('Loan Amount Term', [360.0, 180.0, 120.0, 240.0])
    Credit_History = st.selectbox('Credit History', [0.0, 1.0])
    Property_Area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

    submitted = st.form_submit_button('Predict')

if submitted:
    if model is None or encoders is None:
        st.warning('Model artifacts not found. Run `train_model.py` to generate them.')
    else:
        # Convert LoanAmount to the same unit used during training (thousands)
        unit_factors = {
            'Rupees': 1.0 / 1000.0,
            'Thousands': 1.0,
            'Lakhs': 100.0,    # 1 lakh = 100 thousands
            'Crores': 10000.0  # 1 crore = 10,000 thousands
        }
        LoanAmount_converted = float(LoanAmount) * float(unit_factors.get(LoanAmount_unit, 1.0))

        # Also compute the requested amount in rupees for display
        unit_to_rupees = {
            'Rupees': 1.0,
            'Thousands': 1000.0,
            'Lakhs': 100000.0,
            'Crores': 10000000.0
        }
        rupees_amount = float(LoanAmount) * float(unit_to_rupees.get(LoanAmount_unit, 1.0))

        raw_input = {
            'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount_converted,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area
        }

        # Build input row following training feature order
        if feature_columns is None:
            st.warning('Feature metadata missing. Retrain to generate metadata.')
            feature_columns = list(raw_input.keys())

        row = {}
        for col in feature_columns:
            # take provided value if available, else default to 0
            val = raw_input.get(col, 0)
            # apply label encoding when encoder exists for this column
            if encoders and col in encoders:
                le = encoders[col]
                try:
                    val_enc = le.transform([str(val)])[0]
                except Exception:
                    # unseen label: fallback to 0
                    val_enc = 0
                row[col] = val_enc
            else:
                row[col] = val

        input_df = pd.DataFrame([row], columns=feature_columns)

        proba = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]

        # Map prediction to original target label if available
        target_label = str(pred)
        if encoders and 'Loan_Status' in encoders:
            try:
                target_label = encoders['Loan_Status'].inverse_transform([pred])[0]
            except Exception:
                target_label = str(pred)

        # Determine index of the "approved" class (encoded as 1 during training)
        try:
            pos_index = list(model.classes_).index(1)
        except Exception:
            # fallback: assume the second column corresponds to approved
            pos_index = 1 if len(proba) > 1 else 0

        approved_prob = float(proba[pos_index])
        eligible = (pred == 1)
        eligibility_text = 'Eligible' if eligible else 'Not eligible'

        st.success(f'Eligibility: {eligibility_text} — Prediction: {target_label}')
        st.info(f'Approved probability: {approved_prob:.2%} — Raw probabilities: {proba.round(3).tolist()}')
        # Show requested amount in rupees when eligible
        try:
            st.write(f'Requested amount: ₹{rupees_amount:,.2f}')
        except Exception:
            st.write(f'Requested amount: {rupees_amount}')
