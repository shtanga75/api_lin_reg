from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
from ml_tools import MLModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Загрузка переменных .env
load_dotenv()

# FastAPI приложение
app = FastAPI(
    title="Linear Regression API",
    description="API для предсказаний с использованием модели линейной регрессии",
    version="1.0.0"
)

# Загрузка параметров из переменных окружения
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_INFO_PATH = os.getenv("MODEL_INFO_PATH", "models/model_info.json")

try:
    ml_model = MLModel(MODEL_PATH, MODEL_INFO_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Ошибка при загрузке модели: {e}")


# Pydantic модели
class PredictionRequest(BaseModel):
    """Модель для запроса предсказания."""
    features: List[float] = Field(
        ..., 
        description="Список значений независимых переменных"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.5, 2.3, -0.5, 1.0]
            }
        }


class PredictionResponse(BaseModel):
    """Модель для ответа с предсказанием."""
    prediction: float = Field(..., description="Предсказанное значение")
    feature_names: List[str] = Field(..., description="Названия используемых признаков")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 42.5,
                "feature_names": ["x1", "x2", "x3", "x4"]
            }
        }


class ModelInfo(BaseModel):
    """Модель для информации о модели."""
    coefficients: List[float] = Field(..., description="Коэффициенты регрессии")
    intercept: float = Field(..., description="Свободный член (intercept)")
    r2_score: float = Field(..., description="R² на тестовой выборке")
    feature_names: List[str] = Field(..., description="Названия признаков")
    
    class Config:
        json_schema_extra = {
            "example": {
                "coefficients": [10.5, 20.3, -5.2, 15.1],
                "intercept": 5.0,
                "r2_score": 0.95,
                "feature_names": ["x1", "x2", "x3", "x4"]
            }
        }


class StatusResponse(BaseModel):
    """Модель для ответа о статусе."""
    status: str = Field(..., description="Статус сервера")


# Создание роутера для API v1 с префиксом
api_v1_router = APIRouter(prefix="/api/v1", tags=["API v1"])


# ЭНДПОИНТЫ

@app.get("/ping", response_model=StatusResponse, tags=["Health"])
def ping():
    """
    Проверка работоспособности сервера.
    
    Returns:
        StatusResponse: Статус сервера
    """
    return {"status": "ok"}


@api_v1_router.post("/prediction", response_model=PredictionResponse, tags=["Prediction"])
def predict_post(request: PredictionRequest):
    """
    Выполнить предсказание (POST запрос).
    
    Args:
        request: Объект с признаками для предсказания
        
    Returns:
        PredictionResponse: Результат предсказания
        
    Raises:
        HTTPException: Если данные некорректны
    """
    try:
        prediction = ml_model.predict(request.features)
        return {
            "prediction": prediction,
            "feature_names": ml_model.get_feature_names()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_v1_router.get("/prediction", response_model=PredictionResponse, tags=["Prediction"])
def predict_get(features: List[float]):
    """
    Выполнить предсказание (GET запрос).
    
    Args:
        features: Список значений признаков (например: ?features=1.5&features=2.3&features=-0.5&features=1.0)
        
    Returns:
        PredictionResponse: Результат предсказания
        
    Raises:
        HTTPException: Если данные некорректны
    """
    try:
        prediction = ml_model.predict(features)
        return {
            "prediction": prediction,
            "feature_names": ml_model.get_feature_names()
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Ошибка при обработке признаков: {str(e)}"
        )


@api_v1_router.get("/model_info", response_model=ModelInfo, tags=["Model Info"])
def get_model_info():
    """
    Получить информацию о модели линейной регрессии.
    
    Returns:
        ModelInfo: Коэффициенты, intercept и R² score
    """
    return {
        "coefficients": ml_model.get_coefficients(),
        "intercept": ml_model.get_intercept(),
        "r2_score": ml_model.get_r2_score(),
        "feature_names": ml_model.get_feature_names()
    }


# Подключение роутера к приложению
app.include_router(api_v1_router)


# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
