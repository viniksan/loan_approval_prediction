# Loan Approval Prediction

Simple project to train a loan approval classifier and serve it via Streamlit.

Structure

- `app.py` — Streamlit web app
- `train_model.py` — training script that saves `loan_model.pkl` and `label_encoders.pkl`
- `requirements.txt` — Python dependencies
- `dataset/loan_data.csv` — sample dataset

Quick start

1. Create and activate a virtual environment (Windows):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model:

```powershell
python train_model.py
```

This creates `loan_model.pkl` and `label_encoders.pkl`.

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes
- If you have your own dataset, replace `dataset/loan_data.csv` with your file. Make sure column names match those used in `train_model.py`.
