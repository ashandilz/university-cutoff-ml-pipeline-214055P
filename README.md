# Sri Lankan UGC University Cutoff Mark Predictor

This project is an end-to-end Machine Learning pipeline designed to predict the A/L Z-Score cutoffs for various university courses in Sri Lanka.

## Project Structure
- `data_prep.py`: Loads the dataset from Hugging Face, cleans the data, and prepares it for training.
- `train.py`: Trains a RandomForestRegressor with hyperparameter tuning and generates SHAP explanations.
- `app.py`: A premium Streamlit dashboard for end-user predictions.
- `models/`: Directory containing saved model, encoders, and plots (generated after running scripts).
- `requirements.txt`: List of Python dependencies.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python data_prep.py
```

### 3. Train the Model
```bash
python train.py
```

### 4. Launch the Web App
```bash
streamlit run app.py
```

## Features
- **Data Source**: [Hugging Face Dataset](https://huggingface.co/datasets/kasi-ranaweera/Sri_Lankan_UGC_Cutoff_Mark_Dataset)
- **Model**: RandomForestRegressor with GridSearchCV tuning.
- **Explainability**: SHAP (SHapley Additive exPlanations) for local and global model transparency.
- **UI**: Modern Streamlit interface with interactive inputs and visualizations.