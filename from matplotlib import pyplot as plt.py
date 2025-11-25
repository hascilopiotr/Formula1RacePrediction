import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
session_q =  fastf1.get_session(2025, 'Mexico', 'Q')
session_q1 = fastf1.get_session(2024, 'Mexico', 'Q')
session_fp2 = fastf1.get_session(2025, 'Mexico', 'FP3')
session_q.load()
session_q1.load()
session_fp2.load()

laps_2024 = session_q1.laps[["Driver", "LapTime"]].copy()   
laps_2024.dropna(subset=["LapTime"], inplace=True)

laps_2025 = session_q.laps[["Driver", "LapTime"]].copy()
laps_2025.dropna(subset=["LapTime"], inplace=True)

laps_fp2 = session_fp2.laps[["Driver", "LapTime"]].copy()
laps_fp2.dropna(subset=["LapTime"], inplace=True)


laps_fp2["LapTimeSeconds"] = laps_fp2["LapTime"].dt.total_seconds()
best_laps = laps_fp2.sort_values(by="LapTime").groupby("Driver", as_index=False).first()
best_laps.rename(columns={"LapTime": "FP2_BestLap"}, inplace=True)
best_laps["FP2_BestLap"] = best_laps["FP2_BestLap"].dt.total_seconds()


    
merged = laps_2025.merge(laps_2024, left_on="Driver", right_on="Driver")

merged["LapTime_x"] = merged["LapTime_x"].dt.total_seconds()
merged["LapTime_y"] = merged["LapTime_y"].dt.total_seconds()

merged = merged.merge(best_laps, on="Driver", how="inner")

laps_2025["LapTime"] = laps_2025["LapTime"].dt.total_seconds()


X = merged[["LapTime_x" , "FP2_BestLap"]]
y = merged[["LapTime_y"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

predicted_lap_times = model.predict(X)
merged["PredictedRaceTime"] = predicted_lap_times

best_laps = merged.sort_values(by="PredictedRaceTime").groupby("Driver", as_index=False).first()


best_laps = best_laps.sort_values(by="PredictedRaceTime")
print(best_laps[["Driver", "PredictedRaceTime"]])


