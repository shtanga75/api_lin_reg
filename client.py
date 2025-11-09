"""
client.py взаимодействие с API сервером.
"""

import requests
import json
from typing import List
import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# URL
BASE_URL = f"http://{os.getenv('SERVER_HOST', '127.0.0.1')}:{os.getenv('SERVER_PORT', '8000')}"


def print_separator():
    """Печать разделителя."""
    print("\n" + "="*60 + "\n")


def check_server_health():
    """Проверить работоспособность сервера."""
    print("1️⃣  Проверка работоспособности сервера (/ping)...")
    try:
        response = requests.get(f"{BASE_URL}/ping")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Статус сервера: {data['status']}")
            print(f"   Код ответа: {response.status_code}")
            return True
        else:
            print(f"❌ Ошибка сервера: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Ошибка подключения: Не удаётся подключиться к {BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def get_model_info():
    """Получить информацию о модели."""
    print("2️⃣  Получение информации о модели (/api/v1/model_info)...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/model_info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Информация о модели:")
            print(f"   Коэффициенты: {data['coefficients']}")
            print(f"   Свободный член (intercept): {data['intercept']}")
            print(f"   R² score на тестовой выборке: {data['r2_score']:.4f}")
            print(f"   Названия признаков: {data['feature_names']}")
            return data
        else:
            print(f"❌ Ошибка: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def predict_post(features: List[float]):
    """
    Выполнить предсказание с POST запросом.
    
    Args:
        features: Список значений признаков
    """
    print(f"3️⃣  Предсказание (POST /api/v1/prediction) с признаками: {features}")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/prediction",
            json={"features": features}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Предсказание: {data['prediction']:.4f}")
            print(f"   Использованы признаки: {data['feature_names']}")
            return data['prediction']
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"   Деталь: {response.json().get('detail', 'Неизвестная ошибка')}")
            return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def predict_get(features: List[float]):
    """
    Выполнить предсказание с GET запросом.
    
    Args:
        features: Список значений признаков
    """
    features_str = ",".join(str(f) for f in features)
    print(f"4️⃣  Предсказание (GET /api/v1/prediction) с признаками: {features}")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/prediction",
            params={"features": features_str}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Предсказание: {data['prediction']:.4f}")
            print(f"   Использованы признаки: {data['feature_names']}")
            return data['prediction']
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"   Деталь: {response.json().get('detail', 'Неизвестная ошибка')}")
            return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def main():
    """Главная функция - выполнить несколько тестовых запросов."""
    print("\n" + "="*60)
    print("КЛИЕНТ ДЛЯ API СЕРВЕРА ЛИНЕЙНОЙ РЕГРЕССИИ")
    print("="*60)
    
    #1.Проверка работоспособности
    if not check_server_health():
        print("\nСервер недоступен. Убедитесь, что сервер запущен на", BASE_URL)
        return
    
    print_separator()
    
    #2.Получение информации
    model_info = get_model_info()
    
    print_separator()
    
    #3.Несколько примеров предсказания с POST запросом
    test_cases_post = [
        [1.5, 2.3, -0.5, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, 1.5, 2.0, -0.5]
    ]
    
    for i, features in enumerate(test_cases_post, 1):
        predict_post(features)
        if i < len(test_cases_post):
            print()
    
    print_separator()
    
    #4.Пример предсказания с GET запросом
    predict_get([2.0, -1.5, 0.5, 1.5])
    
    print_separator()
    print(" Все тесты завершены успешно!\n")


if __name__ == "__main__":
    main()
