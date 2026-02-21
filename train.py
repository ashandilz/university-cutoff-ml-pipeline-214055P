import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import os

def train_model():
    print("Loading processed data...")
    if not os.path.exists('data/processed_data.csv'):
        print("Error: processed_data.csv not found. Please run data_prep.py first.")
        return

    df = pd.read_csv('data/processed_data.csv')
    X = df.drop('Zscore', axis=1)
    y = df['Zscore']

    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Hyperparameter Tuning
    print("Starting hyperparameter tuning (GridSearchCV)...")
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # 3. Evaluation
    print("Evaluating model...")
    y_pred = best_rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Regression Metrics ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # Save metrics to console and model
    joblib.dump(best_rf, 'models/model.pkl')
    print("Model saved to 'models/model.pkl'")

    # 4. Explainability - XAI (SHAP)
    print("Generating SHAP explanations...")
    # Use a subset of test data for SHAP to speed up processing if needed
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_test)

    # SHAP Feature Importance (Bar Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('models/shap_importance_bar.png')
    plt.close()

    # SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot (Beeswarm)")
    plt.tight_layout()
    plt.savefig('models/shap_summary_beeswarm.png')
    plt.close()

    print("SHAP plots saved as 'models/shap_importance_bar.png' and 'models/shap_summary_beeswarm.png'.")

if __name__ == "__main__":
    train_model()
