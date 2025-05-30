from sklearn.preprocessing import StandardScaler
import joblib
import os

SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

def save_scaler(scaler):
    joblib.dump(scaler, SCALER_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

def scale_data(scaler, X):
    return scaler.transform(X)

def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    save_scaler(scaler)
    return scaler