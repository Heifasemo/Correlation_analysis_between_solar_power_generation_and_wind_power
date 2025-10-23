import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
Preprocess the 5B plant dataset:
- Read raw CSV (`final_merged.csv`)
- Clamp negative GHI to 0 and set corresponding `power_2C2` to 0
- Create wind direction sine/cosine features
- Export a clean training CSV (`train_data.csv`)
- Fit a StandardScaler on numeric features and print mean/std
"""

# Read raw CSV
df = pd.read_csv("final_merged.csv")

# Clamp negative GHI to 0; set power to 0 for those rows
mask_negative_ghi = df["ghi"] < 0
df.loc[mask_negative_ghi, "ghi"] = 0
df.loc[mask_negative_ghi, "power_2C2"] = 0

# Auxiliary wind features
df["WIND_SIN"] = np.sin(np.radians(df["wind_direction"]))
df["WIND_COS"] = np.cos(np.radians(df["wind_direction"]))

# If you prefer module averaged temperature, uncomment and adapt:
# df["ave_temperature"] = df[[
#     "sjl003_temperature_1",
#     "sjl005_temperature_2",
#     "sjl007_temperature_1"
# ]].mean(axis=1)

# Columns to export (Timestamp kept but not standardized)
features = [
    "Timestamp",            # kept in output, not standardized
    "ave_temperature",
    "ghi",
    "temperature_ambient",
    "wind_speed",
    "WIND_SIN",
    "WIND_COS",
    "power_2C2",
]
df_features = df[features]
df_features.to_csv("train_data.csv", index=False)
print("New CSV generated:", "train_data.csv")

# Numeric features to standardize (exclude target and Timestamp)
numeric_feats = [
    "ave_temperature",
    "ghi",
    "temperature_ambient",
    "wind_speed",
    "WIND_SIN",
    "WIND_COS",
]
X_num = df_features[numeric_feats].astype(np.float32).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

mean = scaler.mean_
std = np.sqrt(scaler.var_)

print("Feature Mean:", mean)
print("Feature Std: ", std)

# (Optional) Save artifacts for inference
# import joblib
# joblib.dump(scaler, "scaler.pkl")
# np.save("X_scaled.npy", X_scaled)
