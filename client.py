import logging
from logging.handlers import RotatingFileHandler
import sys
import time
import pickle
import numpy as np
import open3d as o3d
import laspy
from tasks import upload_laz, process_laz
from celery.result import AsyncResult
import redis


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

# Подключаемся к Redis (для получения меток)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

# Визуализация результата
def visualize_result(file_path, labels):

    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Цвета для классов: фон-красный, дорога-зелёный, здания-синий, машины-жёлтый
    colors = np.zeros((len(points), 3))
    colors[labels == 0] = [1, 0, 0]  # фон
    colors[labels == 1] = [0, 1, 0]        # дорога
    colors[labels == 2] = [0, 0, 1]        # здания
    colors[labels == 3] = [1, 1, 0]        # машины

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Segmented Point Cloud")

def main(file_path):

    upload_task = upload_laz.delay(file_path)
    logger.info(f'ID загруженой задачи: {upload_task.id}')
    while not upload_task.ready():
        time.sleep(1)
    if upload_task.failed():
        logger.error('Ошибка загрузки:', upload_task.info)
        return
    upload_result = upload_task.result
    logger.info('Загрузка завершена:', upload_result)

    # Обработка
    process_task = process_laz.delay(upload_task.id)
    logger.info(f'ID задачи обработки: {process_task.id}')

   # Мониторим прогресс
    while not process_task.ready():
        result = AsyncResult(process_task.id)
        if result.state == 'PROCESSING' and result.info:
            progress = result.info.get('progress')
            message = result.info.get('message')
            eta = result.info.get('eta', 'Оценивается...')
            
            # Добавляем ETA в вывод
            logger.info(f"\Прогресс: {progress}% - {message} | ETA: {eta}   ", end="", flush=True)
            
        time.sleep(2)
    print() # Перенос строки после завершения цикла

    if process_task.failed():
        logger.error('Ошибка обработки:', process_task.info)
        return

    process_result = process_task.result
    logger.info('Обработка завершена:')
    logger.info(f"  Общее количество точек: {process_result['num_points']}")

    stats_raw = process_result['class_stats']
    # Преобразуем строковые ключи в целые числа
    stats = {int(k): v for k, v in stats_raw.items()}
    logger.info(f"  Распределение классов: фон={stats[0]}, дорога={stats[1]}, здание={stats[2]}, транспорт={stats[3]}")

    # Загружаем метки из Redis
    labels_key = f'laz_labels:{process_task.id}'
    labels_data = redis_client.get(labels_key)
    if labels_data:
        labels = pickle.loads(labels_data)
        logger.info(f'Загружены метки из Redis: {labels.shape}')
        # Визуализация
        visualize_result(file_path, labels)
    else:
        logger.info('Метки не найдены в Redis')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.error('Usage: python client.py <path_to_laz>')
        sys.exit(1)
    main(sys.argv[1])