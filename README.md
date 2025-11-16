# API Сервер для Линейной Регрессии

Простой API-сервер на базе FastAPI для выполнения предсказаний с использованием модели линейной регрессии из sklearn.

https://github.com/VetrovSV/Programming/blob/master/plans/ML/tasks/api.md

## Структура проекта
project/

├── app/ 

    ├── server.py # Главный модуль с эндпоинтами API
    
    └── ml_tools.py # Модуль для работы с моделью МО

├── models/
    
    ├── model.joblib # Обученная модель
    
    └── model_info.json # Коэффициенты и метрики

├── .env # Переменные окружения

├── requirements.txt # Зависимости проекта
    
├── README.md # Этот файл
    
└── client.py # Консольный клиент

## Настройка приложения

### 1. Создание виртуального окружения

python -m venv venv
venv\Scripts\activate

### 2. Установка зависимостей

pip install -r requirements.txt

### 3. Подготовка модели

Убедитесь, что модель и информация о ней находятся в папке `models/`:
- `model.joblib` - обученная модель линейной регрессии
- `model_info.json` - файл с коэффициентами и R² score

Для создания модели запустите код `lin_reg.py`.

### 4. Конфигурация .env файла

Создайте файл `.env` в корневой папке проекта:

SERVER_HOST=127.0.0.1
SERVER_PORT=8000
MODEL_PATH=../models/model.joblib
MODEL_INFO_PATH=../models/model_info.json

## Запуск приложения

Запуск сервера
python -m uvicorn app.server:app --reload --host 127.0.0.1 --port 8000

Сервер будет запущен на `http://127.0.0.1:8000`

## Документация API

После запуска сервера документация доступна по адресам:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc

## Проверка работоспособности

### Способ №1: Используя клиент

python client.py
Клиент выполнит несколько тестовых запросов и выведет результаты.

### Способ №2: Используя браузер

#### Проверка сервера

http://127.0.0.1:8000/ping
Ответ:

```
{"status":"ok"}
```



#### Получение информации о модели

http://127.0.0.1:8000/api/v1/model_info

#### Предсказание (POST)

http://127.0.0.1:8000/api/v1/prediction

```
-H "Content-Type: application/json"
-d '{"features": [1.5, 2.3, -0.5, 1.0]}'
```




#### Предсказание (GET)

curl "http://127.0.0.1:8000/api/v1/prediction?features=1.5,2.3,-0.5,1.0"

# Описание эндпоинтов

### GET /ping
Проверка работоспособности сервера.

**Ответ:**

```
{"status": "ok"}
```



### POST /api/v1/prediction
Выполнить предсказание (POST запрос).

**Тело запроса:**

```
{"features": [1.5, 2.3, -0.5, 1.0]}
```

**Ответ:**

```
{
"prediction": 42.5,
"feature_names": ["x1", "x2", "x3", "x4"]
}
```



### GET /api/v1/prediction
Выполнить предсказание (GET запрос).

**Параметры:**
- `features` (строка): значения признаков, разделённые запятыми

**Пример:**

```
/api/v1/prediction?features=1.5,2.3,-0.5,1.0
```

**Ответ:**

```
{
"prediction": 42.5,
"feature_names": ["x1", "x2", "x3", "x4"]
}
```



### GET /api/v1/model_info
Получить информацию о модели.

**Ответ:**

```
{
"coefficients": [10.5, 20.3, -5.2, 15.1],
"intercept": 5.0,
"r2_score": 0.95,
"feature_names": ["x1", "x2", "x3", "x4"]
}
```

