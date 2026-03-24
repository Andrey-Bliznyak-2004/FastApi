import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import uuid
from celery import chain
from tasks import app as celery_app
from tasks import upload_laz, process_laz, generate_visualization_task
from celery.result import AsyncResult
import plotly.graph_objs as go
# Настройка логгирования
logger = logging.getLogger("PointcloudAPI")
logger.setLevel(logging.INFO)

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

# Директории для данных
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI(title="Шлюз сегментатора облаков точек")

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

    logger.info(f"Файл {file.filename} сохранен: {temp_path}")

    # Генерируем уникальный ID для главной задачи обработки (process_laz)
    main_task_id = str(uuid.uuid4())

    # Используем Celery chain для строго последовательного выполнения.
    # Результат первой задачи (метаданные) автоматически передастся во вторую,
    # а результат второй (статистика) — в третью (визуализацию).
    workflow = chain(
        upload_laz.s(temp_path),
        process_laz.s().set(task_id=main_task_id),
        generate_visualization_task.s()
    )
    
    # Запускаем цепочку
    workflow.apply_async()

    logger.info(f"Цепочка задач запущена. ID главной задачи: {main_task_id}")
    
    # Возвращаем ID именно процесса сегментации, 
    # чтобы эндпоинт /status мог считывать проценты прогресса
    return {
        "task_id": main_task_id,         
        "status": "PENDING",
        "info": "Обработка запущена"
    }

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": res.state}

    if res.state == 'PENDING':
        response.update({
            "status": "Ожидание",
            "message": "Задача ожидает выполнения"
        })
    elif res.state == 'PROCESSING':
        info = res.info if isinstance(res.info, dict) else {}
        response.update({
            "status": "Обработка",
            "progress": info.get('progress', 0),
            "eta": info.get('eta', "Расчет..."),
            "message": info.get('message', "")
        })
    elif res.state == 'SUCCESS':
        response.update({
            "status": "Готово",
            "result_summary": res.result
        })
    elif res.state == 'FAILURE':
        response.update({
            "status": "Ошибка",
            "error_detail": str(res.info)
        })
    
    return response

@app.get("/download/{task_id}")
async def download_file(task_id: str):
    output_path = os.path.join(RESULT_DIR, f"result_{task_id}.laz")

    if not os.path.exists(output_path):
        raise HTTPException(
            status_code=404,
            detail="Файл результата не найден. Возможно, обработка еще не завершена."
        )

    return FileResponse(
        output_path,
        media_type='application/octet-stream',
        filename=f"segmented_{task_id}.laz"
    )

@app.get("/visualize/{task_id}", response_class=HTMLResponse)
async def visualize_web(task_id: str):
    html_path = os.path.join(RESULT_DIR, f"viz_{task_id}.html")
    
    if not os.path.exists(html_path):
        # Проверяем, готова ли задача
        res = AsyncResult(task_id, app=celery_app)
        if not res.ready():
            raise HTTPException(status_code=202, detail="Визуализация еще подготавливается")
        raise HTTPException(status_code=404, detail="Файл визуализации не найден")

    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)