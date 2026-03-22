import sys
import time
import pickle
import numpy as np
import open3d as o3d
import laspy
from tasks import upload_laz, process_laz
from celery.result import AsyncResult
import redis

# Подключаемся к Redis (для получения меток)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

def visualize_result(file_path, labels):
    """Визуализирует облако точек с раскраской по классам."""
    # Читаем точки из файла
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Цвета для классов: фон-серый, дорога-зелёный, здания-синий, машины-красный
    colors = np.zeros((len(points), 3))
    colors[labels == 0] = [1, 0, 0]  # фон
    colors[labels == 1] = [0, 1, 0]        # дорога
    colors[labels == 2] = [0, 0, 1]        # здания
    colors[labels == 3] = [1, 1, 0]        # машины

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Segmented Point Cloud")

def main(file_path):
    # Шаг 1: загрузка
    upload_task = upload_laz.delay(file_path)
    print(f'Upload task ID: {upload_task.id}')
    while not upload_task.ready():
        time.sleep(1)
    if upload_task.failed():
        print('Upload failed:', upload_task.info)
        return
    upload_result = upload_task.result
    print('Upload completed:', upload_result)

    # Шаг 2: обработка
    process_task = process_laz.delay(upload_task.id)
    print(f'Process task ID: {process_task.id}')

    # Мониторим прогресс
   # Мониторим прогресс
    while not process_task.ready():
        result = AsyncResult(process_task.id)
        if result.state == 'PROCESSING' and result.info:
            progress = result.info.get('progress')
            message = result.info.get('message')
            eta = result.info.get('eta', 'Оценивается...') # Забираем ETA
            
            # Добавляем ETA в вывод (используем \r чтобы строка обновлялась на месте, а не спамила в консоль)
            print(f"\rProgress: {progress}% - {message} | ETA: {eta}   ", end="", flush=True)
            
        time.sleep(2)
    print() # Перенос строки после завершения цикла

    if process_task.failed():
        print('Process failed:', process_task.info)
        return

    process_result = process_task.result
    print('Process completed:')
    print(f"  Total points: {process_result['num_points']}")

    stats_raw = process_result['class_stats']
    # Преобразуем строковые ключи в целые числа
    stats = {int(k): v for k, v in stats_raw.items()}
    print(f"  Class distribution: background={stats[0]}, road={stats[1]}, building={stats[2]}, vehicles={stats[3]}")

    # Загружаем метки из Redis
    labels_key = f'laz_labels:{process_task.id}'
    labels_data = redis_client.get(labels_key)
    if labels_data:
        labels = pickle.loads(labels_data)
        print(f'Loaded labels from Redis, shape: {labels.shape}')
        # Визуализация
        visualize_result(file_path, labels)
    else:
        print('No labels found in Redis')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python client.py <path_to_laz>')
        sys.exit(1)
    main(sys.argv[1])