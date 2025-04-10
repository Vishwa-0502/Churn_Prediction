import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model_and_scaler():
    """Create a realistic model for bank customer churn prediction"""
    try:
        logging.info("Creating realistic model and scaler...")
        
        # Create models
        lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        rf_model = RandomForestClassifier(n_estimators=50, class_weight='balanced')
        
        # Build ensemble
        model = VotingClassifier(
            estimators=[
                ('lr', lr_model),
                ('rf', rf_model)
            ],
            voting='soft'
        )
        
        # Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, 
        # HasCrCard, IsActiveMember, EstimatedSalary, BalanceSalaryRatio, TenureByAge, CreditScorePerAge
        
        n_samples = 1000
        X_dummy = np.zeros((n_samples, 13))
        
        # Generate more realistic features
        X_dummy[:, 0] = np.random.randint(400, 850, n_samples)  # CreditScore
        X_dummy[:, 1] = np.random.randint(0, 3, n_samples)      # Geography
        X_dummy[:, 2] = np.random.randint(0, 2, n_samples)      # Gender
        X_dummy[:, 3] = np.random.randint(18, 95, n_samples)    # Age
        X_dummy[:, 4] = np.random.randint(0, 11, n_samples)     # Tenure
        
        # Balances - some zero, some very high
        balances = np.zeros(n_samples)
        has_balance = np.random.rand(n_samples) < 0.8  # 80% have balance
        balances[has_balance] = np.random.exponential(50000, sum(has_balance))
        X_dummy[:, 5] = balances
        
        X_dummy[:, 6] = np.random.randint(1, 5, n_samples)      # NumOfProducts
        X_dummy[:, 7] = np.random.randint(0, 2, n_samples)      # HasCrCard
        X_dummy[:, 8] = np.random.randint(0, 2, n_samples)      # IsActiveMember
        X_dummy[:, 9] = np.random.uniform(10000, 200000, n_samples)  # EstimatedSalary
        
        # Derived features
        X_dummy[:, 10] = X_dummy[:, 5] / (X_dummy[:, 9] + 1)    # BalanceSalaryRatio
        X_dummy[:, 11] = X_dummy[:, 4] / (X_dummy[:, 3] + 1)    # TenureByAge
        X_dummy[:, 12] = X_dummy[:, 0] / (X_dummy[:, 3] + 1)    # CreditScorePerAge
        
        # Generate target variable with dependencies on features
        # More likely to churn if: lower credit score, inactive member, 
        # age extremes, higher # of products, low balance/salary ratio
        churn_prob = np.zeros(n_samples)
        
        # Credit score factor (higher score = lower churn)
        churn_prob += 0.4 * (1 - (X_dummy[:, 0] - 400) / 450)
        
        # Age factor (middle age = lower churn)
        age_factor = 0.5 * np.abs(X_dummy[:, 3] - 45) / 45
        churn_prob += age_factor
        
        # Active member (inactive = higher churn)
        churn_prob += 0.3 * (1 - X_dummy[:, 8])
        
        # Number of products (1 or 2 = lower churn, 3+ = higher churn)
        prod_factor = np.zeros(n_samples)
        prod_factor[X_dummy[:, 6] == 1] = 0.1
        prod_factor[X_dummy[:, 6] == 2] = 0.0
        prod_factor[X_dummy[:, 6] >= 3] = 0.4
        churn_prob += prod_factor
        
        # Balance factor (zero balance = higher churn)
        balance_factor = np.zeros(n_samples)
        balance_factor[X_dummy[:, 5] < 1] = 0.3
        churn_prob += balance_factor
        
        # Normalize and add randomness
        churn_prob = churn_prob / churn_prob.max()
        churn_prob = 0.7 * churn_prob + 0.3 * np.random.rand(n_samples)
        churn_prob = np.clip(churn_prob, 0, 1)
        
        # Final binary target
        y_dummy = (churn_prob > 0.5).astype(int)
        
        # Make sure we have a good class balance (at least 30% churn)
        while np.mean(y_dummy) < 0.3:
            add_churn = np.where(y_dummy == 0)[0][:50]
            y_dummy[add_churn] = 1
        
        model.fit(X_dummy, y_dummy)
        
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(X_dummy)
        
        # Save model and scaler as compatible pickle files
        with open('custom_churn_model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        with open('custom_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f, protocol=4)
        
        logging.info("Model and scaler created and saved successfully")
        return True
    
    except Exception as e:
        logging.error(f"Error creating model and scaler: {str(e)}")
        return False

if __name__ == "__main__":
    create_model_and_scaler()