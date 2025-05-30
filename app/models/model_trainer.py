import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from .model_utils import fit_scaler, scale_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model1.h5')

FEATURE_COLUMNS = [
    'Temperature', 'Rain', 'Wind_Speed', 'Humidity', 'Snow_Depth', 'Snowfall',
    'Weather_Code', 'Holidays_France', "AllergyPeriod", 'Fête',
    'Day', 'Month', 'Is_Weekend',
    'Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday',
    'Weekday_Thursday', 'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday'
]
#

# Your architecture
def build_model_v2(input_dim):
    model1 = models.Sequential([
        layers.GaussianNoise(0.01, input_shape=(input_dim,)),

        layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model1.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model1

def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["Patients_Count"]
    return X, y

def train_and_save_model(X, y):
    # Scale
    scaler = fit_scaler(X)
    X_scaled = scale_data(scaler, X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    # Build and train
    model = build_model_v2(X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, verbose=1)

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=0  # Optional: change to 1 for logs
    )

    # Evaluate (optional logging)
    preds = model.predict(X_test).flatten()
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R²:", r2_score(y_test, preds))

    # Save
    model.save(MODEL_PATH)