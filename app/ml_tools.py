"""
Модуль ml_tools.py одержит функции загрузки модели и выполнения предсказаний.
"""

import joblib
import json
from typing import List, Dict
import os


class MLModel:
    """Класс для работы с обученной моделью линейной регрессии."""
    
    def __init__(self, model_path: str, info_path: str):
        """
        Инициализация модели.
        
        Args:
            model_path: Путь к файлу модели (format: .joblib)
            info_path: Путь к файлу с информацией о модели (model_info.json)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Информация о модели не найдена: {info_path}")
        
        # Загрузка модели
        self.model = joblib.load(model_path)
        
        # Загрузка информации о модели
        with open(info_path, 'r') as f:
            self.info = json.load(f)
    
    def predict(self, features: List[float]) -> float:
        """
        Выполнить предсказание.
        
        Args:
            features: Список значений независимых переменных
            
        Returns:
            Предсказанное значение целевой переменной
        """
        if len(features) != self.info['n_features']:
            raise ValueError(
                f"Ожидается {self.info['n_features']} признаков, "
                f"получено {len(features)}"
            )
    
        import numpy as np
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        
        return float(prediction)
    
    def get_coefficients(self) -> List[float]:
        """Получить коэффициенты модели."""
        return self.info['coefficients']
    
    def get_intercept(self) -> float:
        """Получить свободный член модели."""
        return self.info['intercept']
    
    def get_r2_score(self) -> float:
        """Получить значение R² на тестовой выборке."""
        return self.info['r2_score']
    
    def get_feature_names(self) -> List[str]:
        """Получить названия признаков модели."""
        return self.info['feature_names']
