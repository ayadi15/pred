from pydantic import BaseModel
from typing import Optional

class InputData(BaseModel):
    Temperature: float
    Rain: float
    Wind_Speed: float
    Humidity: float
    Snow_Depth: float
    Snowfall: float
    Weather_Code: float
    Holidays_France: int
    AllergyPeriod: int
    Fête: int
    Day: int
    Month: int
    Is_Weekend: int
    Weekday_Monday: int
    Weekday_Tuesday: int
    Weekday_Wednesday: int
    Weekday_Thursday: int
    Weekday_Friday: int
    Weekday_Saturday: int
    Weekday_Sunday: int
    Patients_Count: Optional[float] = None  # Will be predicted if not provided