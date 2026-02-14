
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import pickle
import os

def preprocess_data(df):
    """Preprocess the heart disease dataset"""
    # Drop id and dataset columns
    df = df.drop(['id', 'dataset'], axis=1, errors='ignore')

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Convert target to binary (0: no disease, 1: disease)
    if 'num' in df.columns:
        df['num'] = (df['num'] > 0).astype(int)

    return df, label_encoders

def train_knn_model(data_path='data/heart_disease_uci.csv'):
    """Train KNN model and save it"""
    # Load data
    df = pd.read_csv(data_path)

    # Preprocess
    df, label_encoders = preprocess_data(df)

    # Separate features and target
    X = df.drop('num', axis=1)
    y = df['num']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }

    # Save model
    os.makedirs('models_pkl', exist_ok=True)
    with open('models_pkl/knn_model.pkl', 'wb') as f:
        pickle.dump({
            'pipeline': pipeline,
            'label_encoders': label_encoders,
            'feature_names': X.columns.tolist(),
            'metrics': metrics
        }, f)

    print("KNN Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return pipeline, metrics, X_test, y_test, y_pred

if __name__ == "__main__":
    train_knn_model()