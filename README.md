# Customer Churn Prediction Web App

This is a web-based machine learning application that predicts customer churn using a trained ensemble model (Random Forest + Logistic Regression). It supports both **single customer input** and **CSV batch uploads**, with explanations for each prediction.

---

## Features

- Predict churn for individual customers
- Upload CSVs for batch predictions
- View churn probability and risk explanations
- Summary analytics for uploaded datasets
- Custom-trained ensemble model with feature engineering
- Logs and error handling included

---

## Technologies Used

- Python, Flask, HTML/CSS (Jinja2)
- Scikit-learn, Pandas, NumPy
- Bootstrap (via templates)
- Logging, Sessions, and secure file handling
- Pretrained models: `custom_churn_model.pkl` and `custom_scaler.pkl`

---

## Project Structure

```
├── app.py                   # Flask routes + prediction logic
├── main.py                  # Entry point
├── prediction.py            # Model loading, data prep, feature engineering
├── custom_churn_model.pkl   # Pretrained ensemble model (RandomForest + LogisticRegression)
├── custom_scaler.pkl        # StandardScaler for preprocessing
├── templates/
│   ├── index.html           # Home form
│   ├── result.html          # Individual result page
│   └── batch_result.html    # Batch CSV results
├── static/                  # (Optional) CSS, JS for UI
├── uploads/                 # Temp uploaded CSVs
├── notebook.ipynb           # Jupyter training code (optional)
└── README.md
```

---

## Local Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Vishwa-0502/demosite.git
cd demosite
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install flask pandas numpy scikit-learn
```

### 4. Run the App

```bash
python main.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## Example CSV Format for Batch Prediction

```csv
CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
600,0,1,35,5,120000,2,1,1,85000
450,2,0,29,2,0,1,0,0,54000
...
```

---

## Model Details

- **Type**: Voting Classifier (RandomForest + LogisticRegression)
- **Features Used**: 13, including 3 engineered
- **Training Strategy**:
  - Custom synthetic dataset generation
  - Balanced churn ratios (~30%)
  - Saved using `pickle`

---

## Security Notes

- Max upload size: 16 MB
- Only `.csv` files are accepted
- Inputs are validated and errors are flashed
- Secure filenames and Flask session management are implemented

---

## Future Improvements

- Add TensorFlow/Keras ANN support
- User authentication
- Upload and visualize charts using Plotly
- Dockerize app for deployment

---
