import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
import uvicorn
from tasks import app as celery_app
from tasks import upload_laz, process_laz
from celery.result import AsyncResult
import plotly.graph_objs as go
import numpy as np
import laspy
# Настройка логгирования
logger = logging.getLogger("PointcloudAPI")
logger.setLevel(logging.INFO)

# Логгирование в файл с ротацией
file_handler = RotatingFileHandler(
    "app.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Директории для загрузки и хранения результатов
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Инициализация FastAPI
app = FastAPI(title="Шлюз сегментатора облаков точек")

# Маршрут для загрузки и обработки файла 
@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.las', '.laz')):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только файлы .las или .laz"
        )

    temp_path = os.path.join(UPLOAD_DIR, f"raw_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Файл {file.filename} успешно сохранён по пути: {temp_path}")
    logger.info("Запуск цепочки задач Celery (upload_laz → process_laz)")

    # Сначала загружаем файл, затем запускаем обработку 
    chain = (upload_laz.s(temp_path) | process_laz.s())
    result = chain.apply_async()

    logger.info(f"Задача Celery успешно поставлена в очередь. ID: {result.id}")

    # Возвращаем ID задачи и статус
    return {
        "task_id": result.id,
        "status": "в_очереди",
        "info": "Файл принят и поставлен в очередь на обработку"
    }

# Маршрут для получения статуса задачи по айди 
@app.get("/status/{task_id}")
async def get_status(task_id: str):
    logger.info(f"Получен запрос статуса для задачи: {task_id}")

    # Получаем результат из Celery
    res = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": res.state}

    # Обработка разных состояний типа ожидания, обработки и готовнисти
    if res.state == 'PENDING':
        response["status"] = "Ожидание"
        response["message"] = "Задача ожидает запуска в очереди"
    elif res.state == 'PROCESSING':
        response["progress"] = res.info.get('progress') if isinstance(res.info, dict) else None
        response["message"] = res.info.get('message') if isinstance(res.info, dict) else "Обработка в процессе"
        response["eta"] = res.info.get('eta') if isinstance(res.info, dict) else "Оценивается..." 
    elif res.state == 'SUCCESS':
        response["status"] = "Готово"
        response["result_summary"] = res.result
    elif res.state == 'FAILURE':
        response["status"] = "Ошибка"
        response["error_detail"] = str(res.info)
    else:
        response["status"] = "неизвестно"
        response["message"] = "Неизвестное состояние задачи"

    return response

# Скачивание результата
@app.get("/download/{task_id}")
async def download_file(task_id: str):
    logger.info(f"Получен запрос на скачивание результата для задачи: {task_id}")

    # Получаем результат из Celery
    res = AsyncResult(task_id, app=celery_app)

    if not res.ready():
        raise HTTPException(
            status_code=400,
            detail="Задача всё ещё обрабатывается или не найдена"
        )

    if res.failed():
        raise HTTPException(
            status_code=500,
            detail="Задача завершилась с ошибкой"
        )

    output_path = os.path.join(RESULT_DIR, f"result_{task_id}.laz")

    if not os.path.exists(output_path):
        logger.error(f"Файл результата {output_path} не найден на диске!")
        raise HTTPException(
            status_code=404,
            detail="Файл результата отсутствует на сервере"
        )

    logger.info(f"Файл результата найден. Начинаем отправку: {output_path}")

    return FileResponse(
        output_path,
        media_type='application/octet-stream',
        filename=f"segmented_{task_id}.laz"
    )
@app.get("/visualize/{task_id}", response_class=HTMLResponse)
async def visualize_web(task_id: str):
    output_path = os.path.join(RESULT_DIR, f"result_{task_id}.laz")
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Файл результата не найден")

    # Чтение данных
    las = laspy.read(output_path) 
    step = 1
    x, y, z = las.x[::step], las.y[::step], las.z[::step]
    
    # Извлечение цветов, если они есть
    colors = None
    if hasattr(las, 'classification'):
        labels = las.classification[::step]
        # Простая цветовая мапа для примера
        colors = ['red' if l == 0 else 'green' if l == 1 else 'blue' for l in labels]

    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color=colors)
    )
    
    fig = go.Figure(data=[trace])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    
    return fig.to_html(full_html=True, include_plotlyjs='cdn')
    
def cleanup_files(temp_path: str):
    if os.path.exists(temp_path):
        os.remove(temp_path)
        logger.info(f"Временный файл {temp_path} успешно удалён")

# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)