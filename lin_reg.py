from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import joblib
import os
import json

#Генерация данных
X, y = make_regression(n_samples=500, n_features=4, n_informative=2, random_state=212862)
data0 = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2], 'x4': X[:, 3], 'y': y})

#train test
train, test = train_test_split(data0, shuffle=True, random_state=0, test_size=0.2)

#Обучение
lin_reg = LinearRegression()
lin_reg.fit(train[['x1', 'x2', 'x3', 'x4']], train['y'])

#Получение метрик на тестовой выборке
y_pred_test = lin_reg.predict(test[['x1', 'x2', 'x3', 'x4']])
r2_test = r2_score(test['y'], y_pred_test)

#Сохранение модели и метрик
os.makedirs("models", exist_ok=True)
joblib.dump(lin_reg, "models/model.joblib")

#Сохранение параметров модели в JSON
model_info = {
    "coefficients": lin_reg.coef_.tolist(),
    "intercept": float(lin_reg.intercept_),
    "r2_score": float(r2_test),
    "n_features": 4,
    "feature_names": ["x1", "x2", "x3", "x4"]
}

with open("models/model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)
