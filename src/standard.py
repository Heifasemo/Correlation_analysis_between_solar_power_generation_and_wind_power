import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取原始CSV数据
df = pd.read_csv('final_merged.csv')

# 把 ghi 小于 0 的置为 0，同时把对应行的 power_2C2 也置为 0
mask_negative_ghi = df['ghi'] < 0
df.loc[mask_negative_ghi, 'ghi'] = 0
df.loc[mask_negative_ghi, 'power_2C2'] = 0

# 生成辅助列
df['WIND_SIN'] = np.sin(np.radians(df['wind_direction']))
df['WIND_COS'] = np.cos(np.radians(df['wind_direction']))
#df['ave_temperature'] = df[[
#    'sjl003_temperature_1',
#    'sjl005_temperature_2',
#    'sjl007_temperature_1'
#]].mean(axis=1)

# 选出要写入 CSV 的列
features = [
    'Timestamp',   # 保留到输出文件中，但不标准化
    'ave_temperature',
    'ghi',
    'temperature_ambient',
    'wind_speed',
    'WIND_SIN',
    'WIND_COS',
    'power_2C2'
]
df_features = df[features]
df_features.to_csv('train_data.csv', index=False)
print("新CSV文件已生成:", 'train_data.csv')

# 只对数值特征做标准化
numeric_feats = [
    'ave_temperature',
    'ghi',
    'temperature_ambient',
    'wind_speed',
    'WIND_SIN',
    'WIND_COS'
]
X_num = df_features[numeric_feats].astype(np.float32).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

mean = scaler.mean_
std  = np.sqrt(scaler.var_)

print("Feature Mean:", mean)
print("Feature Std: ", std)
