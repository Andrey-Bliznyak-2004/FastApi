import os
import pickle
import redis
from celery import Celery
from celery.utils.log import get_task_logger
from utils import read_las, segment_point_cloud, save_segmented_las
import time
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
logger = get_task_logger(__name__)

# Клиент для Redis (бинарный и строковый)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
redis_str = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

@app.task(bind=True, name='upload_laz')
def upload_laz(self, file_path):
    task_id = self.request.id
    logger.info(f'Upload task started: {task_id}')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    # ВАЖНО: загружаем только метаданные (load_points=False), чтобы не тратить память
    _, _, metadata = read_las(file_path, load_points=False) 
    
    redis_str.setex(f'laz_path:{task_id}', 3600, file_path)
    
    # Добавляем ID задачи в результат, чтобы следующая задача могла его найти
    metadata['source_task_id'] = task_id 
    return metadata

@app.task(bind=True, name='process_laz')
def process_laz(self, upload_data): # Переименовали аргумент для ясности
    task_id = self.request.id
    logger.info(f'Process task started: {task_id}')

    # ПРОВЕРКА: если нам прислали словарь (результат chain), берем ID из него
    if isinstance(upload_data, dict):
        upload_task_id = upload_data.get('source_task_id')
    else:
        upload_task_id = upload_data

    file_path = redis_str.get(f'laz_path:{upload_task_id}')
    if not file_path:
        raise ValueError(f'No file path found for upload task {upload_task_id}')

    points, rgb, metadata = read_las(file_path, load_points=True)

    # Колбэк с расчетом времени (ETA)
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
    # НОВЫЙ КОД: Сохраняем результат в файл
    output_filename = f"result_{task_id}.laz"
    output_path = os.path.join("results", output_filename)
    
    # Создаем папку results если её нет
    os.makedirs("results", exist_ok=True)
    
    # Сохраняем сегментированное облако
    save_segmented_las(file_path, points, rgb, labels, output_path)
    logger.info(f"Результат сохранён в файл: {output_path}")


    # Дополняем статистику недостающими классами (0,1,2,3)
    full_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    full_stats.update(class_stats)
    class_stats = full_stats  # теперь словарь содержит все ключи

    redis_client.setex(f'laz_labels:{task_id}', 3600, pickle.dumps(labels))

    logger.info(f'Class distribution: background={class_stats[0]}, road={class_stats[1]}, building={class_stats[2]}, vehicles={class_stats[3]}')

    result = {
        'task_id': task_id,
        'num_points': len(points),
        'class_stats': class_stats,  # теперь безопасно использовать индексы
        'metadata': metadata,
        'message': 'Segmentation finished'
    }
    return result