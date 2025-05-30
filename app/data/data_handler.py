import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")

def load_dataset():
    return pd.read_csv(DATA_PATH)

def save_dataset(df):
    df.to_csv(DATA_PATH, index=False)

def append_data(new_data: dict):
    df = load_dataset()
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    save_dataset(df)
    return df