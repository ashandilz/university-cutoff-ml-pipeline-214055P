import pandas as pd
import numpy as np
from datasets import load_dataset
import joblib
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import os

def preprocess_data():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("kasi-ranaweera/Sri_Lankan_UGC_Cutoff_Mark_Dataset")
    df = pd.DataFrame(dataset['train'])

    print("Initial data shape:", df.shape)

    # 1. Target Variable: Zscore
    # Convert string to float, handle invalid characters (coerce to NaN then fill/drop)
    df['Zscore'] = pd.to_numeric(df['Zscore'], errors='coerce')
    
    # Drop rows where target is missing or invalid
    df = df.dropna(subset=['Zscore'])
    
    # 2. Feature Selection
    features = ['Exam Year', 'District', 'University', 'Course', 'Stream', 'Intake']
    target = 'Zscore'
    
    # Drop Matched_Course_University to avoid data leakage (as requested)
    # Also ensuring we only keep relevant columns
    df = df[features + [target]]

    # 3. Handling Missing Values for features
    # (Optional but good practice)
    for col in features:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())

    # 4. Encoding
    # High cardinality variables: District, University, Course
    categorical_features = ['District', 'University', 'Course', 'Stream']
    
    print("Encoding categorical features...")
    encoder = TargetEncoder(cols=categorical_features)
    df_encoded = encoder.fit_transform(df[features], df[target])
    
    # 5. Normalization
    print("Normalizing numerical features...")
    scaler = StandardScaler()
    # Exam Year and Intake are numerical
    df_encoded = pd.DataFrame(scaler.fit_transform(df_encoded), columns=features)

    # Save processed data and objects
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    df_processed = pd.concat([df_encoded, df[target].reset_index(drop=True)], axis=1)
    df_processed.to_csv('data/processed_data.csv', index=False)
    
    # We also need original categories for the Streamlit dropdowns
    metadata = {
        'districts': sorted(df['District'].unique().tolist()),
        'universities': sorted(df['University'].unique().tolist()),
        'courses': sorted(df['Course'].unique().tolist()),
        'streams': sorted(df['Stream'].unique().tolist()),
        'years': sorted(df['Exam Year'].unique().tolist()),
        'intakes': sorted(df['Intake'].unique().tolist())
    }
    
    joblib.dump(encoder, 'models/encoder.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(metadata, 'models/metadata.pkl')
    joblib.dump(features, 'models/feature_names.pkl')

    print("Data preprocessing complete. Files saved in 'data/' and 'models/' directories.")
    return df_processed

if __name__ == "__main__":
    preprocess_data()
