import logging
from logging.handlers import RotatingFileHandler
import os
import pickle
import redis
from celery import Celery
from celery.utils.log import get_task_logger
from utils import read_las, segment_point_cloud, save_segmented_las
import time


REDIS_PORT = 6379
REDIS_DB = 0
REDIS_HOST = os.getenv('REDIS_HOST', 'redis') 
REDIS_URL = f'redis://{REDIS_HOST}:6379/0'
app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
logger = logging.getLogger("ClientLog")
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

# Клиент для Redis 
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
redis_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

@app.task(bind=True, name='upload_laz')
def upload_laz(self, file_path):
    task_id = self.request.id
    logger.info(f'Upload task started: {task_id}')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    # Загружаем только метаданные
    _, _, metadata = read_las(file_path, load_points=False) 
    
    redis_str.setex(f'laz_path:{task_id}', 3600, file_path)
    
    # Добавляем ID задачи в результат, чтобы следующая задача могла его найти
    metadata['source_task_id'] = task_id 
    return metadata

@app.task(bind=True, name='process_laz')
def process_laz(self, upload_data): # Переименовали аргумент для ясности
    task_id = self.request.id
    logger.info(f'Запущена задача обработки: {task_id}')

    # Если прислали словарь, берем ID из него
    if isinstance(upload_data, dict):
        upload_task_id = upload_data.get('source_task_id')
    else:
        upload_task_id = upload_data

    file_path = redis_str.get(f'laz_path:{upload_task_id}')
    if not file_path:
        raise ValueError(f'Не найден путь к файлу для задачи {upload_task_id}')

    points, rgb, metadata = read_las(file_path, load_points=True)

    # Расчет времени
    start_time = time.time()
    def progress_callback(processed, total):
        elapsed = time.time() - start_time
        eta_str = "Оценивается..."
        
        if elapsed > 0 and processed > 0:
            speed = processed / elapsed
            remaining = (total - processed) / speed
            mins, secs = divmod(int(remaining), 60)
            eta_str = f'{mins:02d}:{secs:02d}'

        progress = int(10 + 80 * processed / total)
        self.update_state(
            state='PROCESSING', 
            meta={
                'progress': progress, 
                'message': f'{processed}/{total} points',
                'eta': eta_str
            }
        )

    labels, class_stats = segment_point_cloud(points, rgb, progress_callback)
    
    output_filename = f"result_{task_id}.laz"
    output_path = os.path.join("results", output_filename)
    
    # Создаем папку results если её нет
    os.makedirs("results", exist_ok=True)
    
    # Сохраняем сегментированное облако
    save_segmented_las(file_path, points, rgb, labels, output_path)
    logger.info(f"Результат сохранён в файл: {output_path}")


    # Дополняем статистику недостающими классами 
    full_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    full_stats.update(class_stats)
    class_stats = full_stats 

    redis_client.setex(f'laz_labels:{task_id}', 3600, pickle.dumps(labels))

    logger.info(f'Распределение классов: фон={class_stats[0]}, дорога={class_stats[1]}, здание={class_stats[2]}, транспорт={class_stats[3]}')

    result = {
        'task_id': task_id,
        'num_points': len(points),
        'class_stats': class_stats,  
        'metadata': metadata,
        'message': 'Сегментация завершена, результат сохранён в файл: ' + output_path
    }
    return result