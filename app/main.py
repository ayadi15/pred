
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from app.schemas.input_schemas import InputData
# from app.data.data_handler import load_dataset, append_data
# from app.models.model_utils import load_scaler, scale_data
# from app.models.model_trainer import train_and_save_model, preprocess, FEATURE_COLUMNS
# import tensorflow as tf
# import pandas as pd
# import os
# import threading
# from datetime import datetime, timedelta
# import openmeteo_requests
# import pandas as pd
# import requests_cache
# from retry_requests import retry
# from datetime import datetime, date
# import pandas as pd


# # Setup the Open-Meteo client
# cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
# retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
# openmeteo = openmeteo_requests.Client(session=retry_session)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models/model1.h5")
# SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

# app = FastAPI()

# # Globals for model and scaler
# model = None
# scaler = None

# # New Pydantic model for range input
# class DateRangeInput(BaseModel):
#     start_date: str  # format: 'YYYY-MM-DD'
#     end_date: str    # format: 'YYYY-MM-DD'
# selected_features = [
#     'Temperature', 'Rain', 'Wind_Speed', 'Humidity', 'Snow_Depth', 'Snowfall',
#     'Weather_Code', 'Holidays_France', "AllergyPeriod", 'Fête',
#     'Day', 'Month', 'Is_Weekend',
#     'Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday',
#     'Weekday_Thursday', 'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday'
# ]
# def load_model_and_scaler():
#     global model, scaler
#     model = tf.keras.models.load_model(MODEL_PATH)
#     scaler = load_scaler()

# @app.on_event("startup")
# def startup_event():
#     # If model/scaler missing, train from scratch
#     if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
#         df = load_dataset()
#         X, y = preprocess(df)
#         train_and_save_model(X, y)
#     # Load model and scaler globally
#     load_model_and_scaler()

# def retrain_in_background(input_dict: dict):
#     # Append new input data to dataset
#     df = append_data(input_dict)
#     X, y = preprocess(df)
#     train_and_save_model(X, y)
#     # Reload the updated model and scaler after training
#     load_model_and_scaler()

# @app.post("/predict")
# def predict_and_retrain(input: InputData, background_tasks: BackgroundTasks):
#     input_dict = input.dict()
#     # Extract features in right order
#     features = {key: input_dict[key] for key in FEATURE_COLUMNS}
#     X_input = pd.DataFrame([features])

#     # Scale input
#     X_scaled = scale_data(scaler, X_input)
#     # Predict immediately
#     prediction = model.predict(X_scaled).flatten()[0]

#     # Use predicted Patients_Count if not provided
#     input_dict["Patients_Count"] = input_dict.get("Patients_Count") or prediction

#     # Schedule retraining in background — no waiting
#     background_tasks.add_task(retrain_in_background, input_dict)

#     # Return prediction right away
#     return {"predicted_patient_count": round(float(prediction), 2)}


# #--------------------- per time ------------------

# def each_date_between(start_date, end_date):
#     start = datetime.strptime(start_date, "%Y-%m-%d")
#     end = datetime.strptime(end_date, "%Y-%m-%d")
#     while start <= end:
#         yield start
#         start += timedelta(days=1)

# def get_weather_for_day(date_str, latitude=52.52, longitude=13.41):
#     """
#     Fetches weather data for a specific day and returns a daily summary.
#     Parameters:
#         date_str (str): Date in format 'YYYY-MM-DD'
#         latitude (float): Latitude of location
#         longitude (float): Longitude of location
#     Returns:
#         dict: Daily weather summary
#     """
#     if isinstance(date_str, (datetime, date)):
#         date_str = date_str.strftime("%Y-%m-%d")
#     elif isinstance(date_str, str):
#         try:
#             datetime.strptime(date_str, "%Y-%m-%d")
#         except ValueError:
#             raise ValueError("date_str must be in 'YYYY-MM-DD' format")

#     url = "https://archive-api.open-meteo.com/v1/archive"  # <-- FIXED HERE

#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "hourly": [
#             "temperature_2m", "weather_code", "relative_humidity_2m",
#             "rain", "snowfall", "snow_depth", "wind_speed_10m"
#         ],
#         "start_date": date_str,
#         "end_date": date_str,
#         "timezone": "auto"  # optional but good
#     }

#     responses = openmeteo.weather_api(url, params=params)
#     response = responses[0]

#     hourly = response.Hourly()

#     temperature = hourly.Variables(0).ValuesAsNumpy()
#     weather_code = hourly.Variables(1).ValuesAsNumpy()
#     humidity = hourly.Variables(2).ValuesAsNumpy()
#     rain = hourly.Variables(3).ValuesAsNumpy()
#     snowfall = hourly.Variables(4).ValuesAsNumpy()
#     snow_depth = hourly.Variables(5).ValuesAsNumpy()
#     wind_speed = hourly.Variables(6).ValuesAsNumpy()

#     weather_series = pd.Series(weather_code)
#     mode_result = weather_series.mode()
#     most_common_code = int(mode_result.iloc[0]) if not mode_result.empty else -1

#     return {
#         "Temperature": float(round(temperature.mean(), 2)),
#         "Rain": float(round(rain.sum(), 2)),
#         "Wind_Speed": float(round(wind_speed.mean(), 2)),
#         "Humidity": float(round(humidity.mean(), 2)),
#         "Snow_Depth": float(round(snow_depth.mean(), 2)),
#         "Snowfall": float(round(snowfall.sum(), 2)),
#         "Weather_Code": most_common_code
#     }

# def get_calendar_features(date):
#     weekday = date.weekday()
#     day = date.day
#     month = date.month

#     party_events = [
#     # Fêtes organisées
#     (8,5), (8,6),  # Festival Electro Beach
#     (9,15), (9,16),  # Soirée Étudiante
#     (10,31), (11,1),  # Soirée Halloween
#     (11,12),  # Festival de la Bière
#     (12,31), (1,1),  # Nouvel An

#     (5,18), (5,19),  # Festival Rock
#     (6,24), (6,25),  # Saint-Jean
#     (6,28), (6,29),  # Soirée Bodega
#     (7,10), (7,11),  # Soirée Champagne
#     (8,23), (8,24),  # Fête des Vins

#     # Événements de football - Ligue des Champions
#     (9,19), (9,20),(9,21),(9,22),(9,23),
#     (9,24),(9,25),(9,26),(9,27),(9,28),(9,29),
#     (9,30),(10,1),(10,2),(10,3),(10,4),(10,5),
#     (10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(10,13), # Phase de groupes UCL 2023-2024
#     (2,13), (3,13),  # Huitièmes de finale UCL
#     (4,9), (4,10),(4,11),(4,12),(4,13),
#     (4,14),(4,15),(4,16),(4,17),(4,18),(4,19),
#     (4,20),(4,21),(4,22),(4,23),(4,24),(4,25),(4,26),(4,27),(4,28),(4,29),(4,30),
#     (5,1),(5,2),(5,3),(5,4),(5,5),
#     (5,6),(5,7),(5,8),  # Quarts & Demi-finales UCL
#     (6,1),  # Finale UCL

#     # Euro 2024 (haute alcoolémie attendue)
#     (6,14),(6,15),(6,16),(6,17),(6,18),(6,19),
#     (6,20),(6,21),(6,22),(6,23),(6,26),(6,25),(6,26),(6,27),(6,28),(6,29),(6,30),
#     (7,1),(7,2),(7,3),(7,4),(7,5),
#     (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),
#     (7,14), # Euro 2024 Allemagne

#     # Autres événements propices à consommation
#     (2,9), (2,10),  # Soirée Carnaval
#     (4,26), (4,27),  # Spring Break local
#      ]
#     us_holidays = [
#      (1, 1), #New Year's Day (Jour de l'An)
#      (1, 5), #Labour Day (Fête du Travail)
#      (8, 5), #Victory in Europe Day (Victoire 1945)
#      (14, 7), #Bastille Day (Fête Nationale)
#      (15, 8), #Assumption of Mary (Assomption)
#      (1, 11), #All Saints Day (La Toussaint)
#      (11, 11), #Armistice Day (Armistice 1918)
#      (25, 12) #Christmas Day (Noël)
#       ]


#     return {
#         "Holidays_France": int((month, day) in us_holidays),
#         "AllergyPeriod": 1 if date.month in [3,4,5,9,10] else 0,
#         "Fête": int((month, day) in party_events),
#         "Day": date.day,
#         "Month": date.month,
#         "Is_Weekend": int(weekday >= 5),
#         "Weekday_Monday": int(weekday == 0),
#         "Weekday_Tuesday": int(weekday == 1),
#         "Weekday_Wednesday": int(weekday == 2),
#         "Weekday_Thursday": int(weekday == 3),
#         "Weekday_Friday": int(weekday == 4),
#         "Weekday_Saturday": int(weekday == 5),
#         "Weekday_Sunday": int(weekday == 6),


#     }

# def build_model_input(weather_data, calendar_data):
#     # Merge dictionaries and order by your model’s selected features
#     all_data = {**weather_data, **calendar_data}
#     # Print or log the merged dictionary
#     print("Merged all_data:", all_data)
#     return [all_data[feature] for feature in selected_features]

# def predict_total_patients(start_date: str, end_date: str, model, scaler):
#     total = 0
#     for day in each_date_between(start_date, end_date):
#         weather_data = get_weather_for_day(day)
#         calendar_features = get_calendar_features(day)

#         input_row = build_model_input(weather_data, calendar_features)
#         scaled_input = scaler.transform([input_row])
#         prediction = model.predict(scaled_input)[0][0]
#         total += prediction
#     return round(total)

# @app.post("/predict-range")
# def predict_range(input: DateRangeInput):
#     total_predicted = predict_total_patients(
#         input.start_date,
#         input.end_date,
#         model,
#         scaler
#     )
#     return {
#         "start_date": input.start_date,
#         "end_date": input.end_date,
#         "predicted_total_patient_count": total_predicted}


# from fastapi import FastAPI
# from app.schemas.input_schemas import InputData
# from app.data.data_handler import load_dataset, append_data
# from app.models.model_utils import load_scaler, scale_data
# from app.models.model_trainer import train_and_save_model, preprocess, FEATURE_COLUMNS
# import tensorflow as tf
# import pandas as pd
# import os

# MODEL_PATH = os.path.join(os.path.dirname(_file_), "models/model1.h5")
# SCALER_PATH = os.path.join(os.path.dirname(_file_), "models/scaler.pkl")

# app = FastAPI()

# @app.on_event("startup")
# def startup():
#     if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
#         df = load_dataset()
#         X, y = preprocess(df)
#         train_and_save_model(X, y)

# @app.post("/predict")
# def predict_and_retrain(input: InputData):
#     input_dict = input.dict()
#     features = {key: input_dict[key] for key in FEATURE_COLUMNS}
#     X_input = pd.DataFrame([features])

#     model = tf.keras.models.load_model(MODEL_PATH)
#     scaler = load_scaler()
#     X_scaled = scale_data(scaler, X_input)

#     prediction = model.predict(X_scaled).flatten()[0]
#     input_dict["Patients_Count"] = input_dict.get("Patients_Count") or prediction

#     df = append_data(input_dict)
#     X, y = preprocess(df)
#     train_and_save_model(X, y)

#     return {"predicted_patient_count": round(float(prediction), 2)}


# import os
# import threading
# import pandas as pd
# import tensorflow as tf
# from fastapi import FastAPI, BackgroundTasks
# from app.schemas.input_schema import InputData

# from app.data.data_handler import load_dataset, append_data
# from app.models.model_utils import load_scaler, scale_data
# from app.models.model_trainer import train_and_save_model, preprocess, FEATURE_COLUMNS


from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.schemas.input_schemas import InputData
from app.data.data_handler import load_dataset, append_data
from app.models.model_utils import load_scaler, scale_data
from app.models.model_trainer import train_and_save_model, preprocess, FEATURE_COLUMNS
import tensorflow as tf
import pandas as pd
import os
import threading
from datetime import datetime, timedelta
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, date
import pandas as pd
import pickle as pkl
import shap
import joblib
import numpy as np

# Setup the Open-Meteo client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/model1.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

app = FastAPI()

# Globals for model and scaler
model = None
scaler = None
explainer = None 
x_background = None
# New Pydantic model for range input
class DateRangeInput(BaseModel):
    start_date: str  # format: 'YYYY-MM-DD'
    end_date: str    # format: 'YYYY-MM-DD'
selected_features = [
    'Temperature', 'Rain', 'Wind_Speed', 'Humidity', 'Snow_Depth', 'Snowfall',
    'Weather_Code', 'Holidays_France', "AllergyPeriod", 'Fête',
    'Day', 'Month', 'Is_Weekend',
    'Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday',
    'Weekday_Thursday', 'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday'
]
def load_model_and_scaler():
    global model, scaler
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler =joblib.load(f)

# === Scale data ===
def scale_data(scaler, df):
    return scaler.transform(df)


# === SHAP initialization ===
def initialize_shap_explainer(X_train_scaled_sample):
    global explainer
    explainer = shap.Explainer(model, X_train_scaled_sample)

# Load background data for SHAP (this must match your training process)

@app.on_event("startup")
def startup_event():
    # If model/scaler missing, train from scratch
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        df = load_dataset()
        X, y = preprocess(df)
        train_and_save_model(X, y)
    # Load model and scaler globally
    load_model_and_scaler()
    df = load_dataset()  # Replace this with your real function
    X, y = preprocess(df)
    X_train_scaled = scale_data(scaler, X)
    
    # Use first 500 rows as SHAP background
    initialize_shap_explainer(X_train_scaled[:500])

def retrain_in_background(input_dict: dict):
    # Append new input data to dataset
    df = append_data(input_dict)
    X, y = preprocess(df)
    train_and_save_model(X, y)
    # Reload the updated model and scaler after training
    load_model_and_scaler()

@app.post("/predict")
def predict_and_retrain(input: InputData, background_tasks: BackgroundTasks):
    input_dict = input.dict()
    # Extract features in right order
    features = {key: input_dict[key] for key in FEATURE_COLUMNS}
    X_input = pd.DataFrame([features])

    # Scale input
    X_scaled = scale_data(scaler, X_input)
    # Predict immediately
    prediction = model.predict(X_scaled).flatten()[0]

    # Use predicted Patients_Count if not provided
    input_dict["Patients_Count"] = input_dict.get("Patients_Count") or prediction

    # Schedule retraining in background — no waiting
    background_tasks.add_task(retrain_in_background, input_dict)

    # Return prediction right away
    return {"predicted_patient_count": round(float(prediction), 2)}


#--------------------- per time ------------------

def each_date_between(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while start <= end:
        yield start
        start += timedelta(days=1)

def get_weather_for_day(date_str, latitude=52.52, longitude=13.41):
    """
    Fetches weather data for a specific day and returns a daily summary.
    Parameters:
        date_str (str): Date in format 'YYYY-MM-DD'
        latitude (float): Latitude of location
        longitude (float): Longitude of location
    Returns:
        dict: Daily weather summary
    """
    if isinstance(date_str, (datetime, date)):
        date_str = date_str.strftime("%Y-%m-%d")
    elif isinstance(date_str, str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date_str must be in 'YYYY-MM-DD' format")

    url = "https://archive-api.open-meteo.com/v1/archive"  # <-- FIXED HERE

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m", "weather_code", "relative_humidity_2m",
            "rain", "snowfall", "snow_depth", "wind_speed_10m"
        ],
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto"  # optional but good
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()

    temperature = hourly.Variables(0).ValuesAsNumpy()
    weather_code = hourly.Variables(1).ValuesAsNumpy()
    humidity = hourly.Variables(2).ValuesAsNumpy()
    rain = hourly.Variables(3).ValuesAsNumpy()
    snowfall = hourly.Variables(4).ValuesAsNumpy()
    snow_depth = hourly.Variables(5).ValuesAsNumpy()
    wind_speed = hourly.Variables(6).ValuesAsNumpy()

    weather_series = pd.Series(weather_code)
    mode_result = weather_series.mode()
    most_common_code = int(mode_result.iloc[0]) if not mode_result.empty else -1

    return {
        "Temperature": float(round(temperature.mean(), 2)),
        "Rain": float(round(rain.sum(), 2)),
        "Wind_Speed": float(round(wind_speed.mean(), 2)),
        "Humidity": float(round(humidity.mean(), 2)),
        "Snow_Depth": float(round(snow_depth.mean(), 2)),
        "Snowfall": float(round(snowfall.sum(), 2)),
        "Weather_Code": most_common_code
    }

def get_calendar_features(date):
    weekday = date.weekday()
    day = date.day
    month = date.month

    party_events = [
    # Fêtes organisées
    (8,5), (8,6),  # Festival Electro Beach
    (9,15), (9,16),  # Soirée Étudiante
    (10,31), (11,1),  # Soirée Halloween
    (11,12),  # Festival de la Bière
    (12,31), (1,1),  # Nouvel An

    (5,18), (5,19),  # Festival Rock
    (6,24), (6,25),  # Saint-Jean
    (6,28), (6,29),  # Soirée Bodega
    (7,10), (7,11),  # Soirée Champagne
    (8,23), (8,24),  # Fête des Vins

    # Événements de football - Ligue des Champions
    (9,19), (9,20),(9,21),(9,22),(9,23),
    (9,24),(9,25),(9,26),(9,27),(9,28),(9,29),
    (9,30),(10,1),(10,2),(10,3),(10,4),(10,5),
    (10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(10,13), # Phase de groupes UCL 2023-2024
    (2,13), (3,13),  # Huitièmes de finale UCL
    (4,9), (4,10),(4,11),(4,12),(4,13),
    (4,14),(4,15),(4,16),(4,17),(4,18),(4,19),
    (4,20),(4,21),(4,22),(4,23),(4,24),(4,25),(4,26),(4,27),(4,28),(4,29),(4,30),
    (5,1),(5,2),(5,3),(5,4),(5,5),
    (5,6),(5,7),(5,8),  # Quarts & Demi-finales UCL
    (6,1),  # Finale UCL

    # Euro 2024 (haute alcoolémie attendue)
    (6,14),(6,15),(6,16),(6,17),(6,18),(6,19),
    (6,20),(6,21),(6,22),(6,23),(6,26),(6,25),(6,26),(6,27),(6,28),(6,29),(6,30),
    (7,1),(7,2),(7,3),(7,4),(7,5),
    (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),
    (7,14), # Euro 2024 Allemagne

    # Autres événements propices à consommation
    (2,9), (2,10),  # Soirée Carnaval
    (4,26), (4,27),  # Spring Break local
     ]
    us_holidays = [
     (1, 1), #New Year's Day (Jour de l'An)
     (1, 5), #Labour Day (Fête du Travail)
     (8, 5), #Victory in Europe Day (Victoire 1945)
     (14, 7), #Bastille Day (Fête Nationale)
     (15, 8), #Assumption of Mary (Assomption)
     (1, 11), #All Saints Day (La Toussaint)
     (11, 11), #Armistice Day (Armistice 1918)
     (25, 12) #Christmas Day (Noël)
      ]


    return {
        "Holidays_France": int((month, day) in us_holidays),
        "AllergyPeriod": 1 if date.month in [3,4,5,9,10] else 0,
        "Fête": int((month, day) in party_events),
        "Day": date.day,
        "Month": date.month,
        "Is_Weekend": int(weekday >= 5),
        "Weekday_Monday": int(weekday == 0),
        "Weekday_Tuesday": int(weekday == 1),
        "Weekday_Wednesday": int(weekday == 2),
        "Weekday_Thursday": int(weekday == 3),
        "Weekday_Friday": int(weekday == 4),
        "Weekday_Saturday": int(weekday == 5),
        "Weekday_Sunday": int(weekday == 6),


    }

def build_model_input(weather_data, calendar_data):
    # Merge dictionaries and order by your model’s selected features
    all_data = {**weather_data, **calendar_data}
    # Print or log the merged dictionary
    print("Merged all_data:", all_data)
    return [all_data[feature] for feature in selected_features]

def predict_total_patients(start_date: str, end_date: str, model, scaler):
    total = 0
    for day in each_date_between(start_date, end_date):
        weather_data = get_weather_for_day(day)
        calendar_features = get_calendar_features(day)

        input_row = build_model_input(weather_data, calendar_features)
        scaled_input = scaler.transform([input_row])
        prediction = model.predict(scaled_input)[0][0]
        total += prediction
    return round(total)

@app.post("/predict-range")
def predict_range(input: DateRangeInput):
    total_predicted = predict_total_patients(
        input.start_date,
        input.end_date,
        model,
        scaler
    )
    return {
        "start_date": input.start_date,
        "end_date": input.end_date,
        "predicted_total_patient_count": total_predicted}


#----------------------------------- reasons, causes
# === SHAP-based spike explanation ===
def get_top_contributing_features(shap_values, feature_names, top_n=5):
    feature_impact = list(zip(feature_names, shap_values))
    sorted_features = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)
    top = sorted_features[:top_n]
    return [
        (name, float(value), "↑" if value > 0 else "↓") for name, value in top
    ]



def explain_spike(input_features_scaled, feature_names, predicted_count, historical_avg, threshold):
    if predicted_count <= historical_avg + threshold:
        return {
            "spike": False,
            "message": "✅ No spike detected. Prediction is within normal range."
        }

    shap_vals = explainer(input_features_scaled)

    # ✅ Correct extraction of SHAP values
    top_all = get_top_contributing_features(shap_vals[0].values, feature_names)

    top_causes = [
        {
            "feature": feature,
            "impact": round(float(impact), 4),
            "direction": direction
        }
        for feature, impact, direction in top_all
        if float(impact) > 0
    ]

    return {
        "spike": True,
        "predicted_count": round(float(predicted_count), 2),
        "historical_avg": round(float(historical_avg), 2),
        "top_causes": top_causes,
        "message": "📈 Spike detected. Top contributing features identified."
    }




# === Explain spike endpoint ===
@app.post("/explain-spike")
def explain_spike_endpoint(input: InputData):
    input_dict = input.dict()
    features = {key: input_dict[key] for key in FEATURE_COLUMNS}
    X_input = pd.DataFrame([features])

    # Scale input
    X_scaled = scale_data(scaler, X_input)

    # Predict patient count
    predicted_count = model.predict(X_scaled).flatten()[0]

    # Placeholder: historical average and threshold (should be updated with real data)
    historical_avg = 80.0
    threshold = 10.0

    explanation = explain_spike(X_scaled, FEATURE_COLUMNS, predicted_count, historical_avg, threshold)
    return explanation


# --------------------------------------------------WAITING TIME 
# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHONE = os.path.join(BASE_DIR, "models/waiting_time_model.h5")
SCALER_PATHONE = os.path.join(BASE_DIR, "models/scalerOne.save")
SHAP_BG_PATHONE = os.path.join(BASE_DIR, "models/sample_background.npy")


CSV_PATHONE = "data/waitdata.csv"
# MODEL_PATHONE = "waiting_time_model.h5"
# SCALER_PATHONE = "scalerOne.save"
# SHAP_BG_PATHONE = "sample_background.npy"

FEATURE_COLUMNSONE = ["beds_available", "queue", "highpriority", "available_doctors", "available_nurses",
                   "severity_Critical", "severity_Minor", "severity_Moderate", "severity_Severe", "Patient_count"]

# === Load model, scaler, background, SHAP ===
modelOne = tf.keras.models.load_model(MODEL_PATHONE, compile=False)
scalerOne = joblib.load(SCALER_PATHONE)
background_dataOne = np.load(SHAP_BG_PATHONE)
explainerOne = shap.Explainer(modelOne.predict, background_dataOne)

# === Load data for fallback/internal use (optional) ===
try:
    df = pd.read_csv(CSV_PATHONE)
    df = df[FEATURE_COLUMNSONE + ['waiting_time']]
except Exception as e:
    print(f"⚠️ Could not load CSV: {e}")
    df = pd.DataFrame(columns=FEATURE_COLUMNSONE)


# === Input schema ===
class InputDataOne(BaseModel):
    data: dict  # Dict with feature names and values
    avg_wait_time: float = 10.0
    threshold: float = 4.0

# === Utility functions ===
def preprocess_inputOne(input_dict):
    try:
        input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNSONE)
        scaled = scalerOne.transform(input_df)
        return scaled
    except Exception as e:
        raise ValueError(f"Input preprocessing failed: {e}")

def get_top_featuresOne(shap_values_row, top_n=5):
    values = shap_values_row.values.flatten()
    indices = np.abs(values).argsort()[::-1][:top_n]
    return [(FEATURE_COLUMNSONE[i], float(values[i])) for i in indices]

def explain_if_highOne(predicted, scaled_input, avg_wait_time, threshold):
    if predicted <= avg_wait_time + threshold:
        return {
            "high_waiting_time": False,
            "predicted_waiting_time": float(predicted),
            "historical_avg_waiting_time": float(avg_wait_time),
            "message": "✅ Predicted waiting time is within expected range."
        }

    shap_valuesOne = explainerOne(scaled_input)
    top_factorsOne = get_top_featuresOne(shap_valuesOne)

    top_positiveOne = [(name, val) for name, val in top_factorsOne if val > 0]

    return {
        "high_waiting_time": True,
        "predicted_waiting_time": float(predicted),
        "historical_avg_waiting_time": float(avg_wait_time),
        "top_causes": [
            {"feature": name, "impact": value, "direction": "↑"}
            for name, value in top_positiveOne
        ],
        "message": "🚨 High waiting time detected. Top contributing factors identified."
    }

# === API Endpoint ===
@app.post("/predict-waiting-time/")
def predict_waiting_time(payload: InputDataOne):
    input_dictOne = payload.data
    avg_wait_time = payload.avg_wait_time
    thresholdOne = payload.threshold

    missing = [f for f in FEATURE_COLUMNSONE if f not in input_dictOne]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing input features: {missing}")

    try:
        scaled_input = preprocess_inputOne(input_dictOne)
        predictedOne = modelOne.predict(scaled_input).flatten()[0]
        explanationOne = explain_if_highOne(predictedOne, scaled_input, avg_wait_time, thresholdOne)
        return explanationOne
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------CHATBOT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import traceback
from huggingface_hub import InferenceClient



# Hugging Face setup - using a chat-ready model
HF_API_TOKEN = "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"  # ← Replace with env var in production!
llm_client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-0528",  # This supports chat_completion
    token=HF_API_TOKEN
)



class ChatResponse(BaseModel):
    predicted_count: float
    spike: bool
    top_causes: list
    recommendation: str



def get_top_contributing_featuresTwo(shap_vals, feature_names, top_k=5):
    feature_importances = [(name, val) for name, val in zip(feature_names, shap_vals)]
    sorted_importances = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return [(name, float(val), "↑" if val > 0 else "↓") for name, val in sorted_importances if val > 0]

@app.post("/recommending", response_model=ChatResponse)
async def recommend_solutioning(input: InputData):
    try:
        input_dict = input.dict()

        # Filter input_dict to only the features the model expects
        filtered_input = {k: input_dict[k] for k in selected_features if k in input_dict}
        input_df = pd.DataFrame([filtered_input])
        # input_scaled = scaler.transform(input_df)  # Uncomment when real scaler is available

        predicted_count = 100  # Replace with real prediction from model.predict(...)

        historical_avg = 80
        threshold = 10

        if predicted_count <= historical_avg + threshold:
            return {
                "spike": False,
                "predicted_count": round(float(predicted_count), 2),
                "top_causes": [],
                "recommendation": "No spike detected. No action needed."
            }

        # Dummy SHAP values for testing
        shap_vals = [[10.22, -2.3, 5.1]]  # Replace with real SHAP values

        top_causes_raw = get_top_contributing_featuresTwo(shap_vals[0], selected_features)

        top_causes = [
            {
                "feature": name,
                "impact": round(impact, 2),
                "direction": direction
            }
            for name, impact, direction in top_causes_raw
        ]

        # Format prompt for LLM
        cause_lines = "\n".join(
            f"- {cause['feature']} (impact: {cause['impact']}, direction: {cause['direction']})"
            for cause in top_causes
        )
        prompt = f"""
You are an expert hospital operations assistant.

Today there was an unusually high number of patient admissions. The following features contributed most to this spike:

{cause_lines}

Please provide:
1. Likely reasons behind this spike based on the contributing features.
2. Actionable strategies hospital administrators can take to reduce patient load on similar days in the future.

Format your response clearly.
        """

        # ✅ Use chat_completion instead of text_generation
        response = llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=300,
            temperature=0.7,
            top_p=0.9
        )

        chatbot_response = response.choices[0].message.content

        return {
            "spike": True,
            "predicted_count": round(float(predicted_count), 2),
            "top_causes": top_causes,
            "recommendation": chatbot_response
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print("Error occurred:\n", error_details)
        raise HTTPException(status_code=500, detail=str(e))