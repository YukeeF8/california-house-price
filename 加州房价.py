import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

california=fetch_california_housing(as_frame=True)
df=california.data
df['MedHouseValue']=california.target

X=df.drop(columns=['MedHouseValue'])
y=df['MedHouseValue']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

print(f"R² 决定系数：{r2:.4f}")
print(f"平均绝对误差 MAE：{mae:.4f}")

# 造一个房子数据，预测价格
import numpy as np

new_house = [[8.0, 30, 6, 1, 500, 300, 37.8, -122.2]]

new_house_scaled = scaler.transform(new_house)

predict_price = model.predict(new_house_scaled)
print(f"\n预测房价：{predict_price[0]:.2f} (单位：10万美元)")
print(f"折合人民币约：{predict_price[0]*10*7.2:.2f} 万")