import logging
from logging.handlers import RotatingFileHandler
import sys
import time
import webbrowser
from celery.result import AsyncResult

# Импортируем задачи Celery
from tasks import upload_laz, process_laz, visualize_laz

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


def main(file_path):
    # --- Шаг 1: Загрузка метаданных ---
    logger.info('Запуск задачи загрузки...')
    upload_task = upload_laz.delay(file_path)
    logger.info(f'ID загруженной задачи: {upload_task.id}')
    
    while not upload_task.ready():
        time.sleep(1)
        
    if upload_task.failed():
        logger.error(f'Ошибка загрузки: {upload_task.info}')
        return
        
    upload_result = upload_task.result
    logger.info(f'Загрузка завершена: {upload_result}')


    # --- Шаг 2: Обработка (сегментация) ---
    logger.info('Запуск задачи обработки...')
    process_task = process_laz.delay(upload_task.id)
    logger.info(f'ID задачи обработки: {process_task.id}')

    # Мониторим прогресс
    while not process_task.ready():
        result = AsyncResult(process_task.id)
        if result.state == 'PROCESSING' and result.info:
            progress = result.info.get('progress')
            message = result.info.get('message')
            eta = result.info.get('eta', 'Оценивается...')
            
            # Используем \r для перезаписи строки в консоли
            print(f"\rПрогресс: {progress}% - {message} | ETA: {eta}   ", end="", flush=True)
            
        time.sleep(2)
    print()  # Перенос строки после завершения цикла

    if process_task.failed():
        logger.error(f'Ошибка обработки: {process_task.info}')
        return

    process_result = process_task.result
    logger.info('Обработка завершена:')
    logger.info(f"  Общее количество точек: {process_result.get('num_points', 'Неизвестно')}")

    # Извлечение и вывод статистики
    stats_raw = process_result.get('class_stats', {})
    stats = {int(k): v for k, v in stats_raw.items()}
    
    bg = stats.get(0, 0)
    road = stats.get(1, 0)
    building = stats.get(2, 0)
    vehicle = stats.get(3, 0)
    logger.info(f"  Распределение классов: фон={bg}, дорога={road}, здание={building}, транспорт={vehicle}")


    # --- Шаг 3: Визуализация (генерация HTML) ---
    logger.info('Запуск задачи визуализации...')
    visualize_task = visualize_laz.delay(process_result)
    logger.info(f'ID задачи визуализации: {visualize_task.id}')

    while not visualize_task.ready():
        time.sleep(1)

    if visualize_task.failed():
         logger.error(f'Ошибка визуализации: {visualize_task.info}')
         return

    vis_result = visualize_task.result
    html_path = vis_result.get('visualization_path')
    
    if not html_path:
        logger.error('Путь к HTML-файлу не найден в ответе задачи визуализации.')
        return
        
    logger.info(f'Интерактивная визуализация сохранена: {html_path}')
    
    # Открываем результат в браузере по умолчанию
    try:
        logger.info('Открываем визуализацию в браузере...')
        webbrowser.open(html_path)
    except Exception as e:
        logger.error(f"Не удалось открыть браузер автоматически: {e}. Вы можете открыть файл {html_path} вручную.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.error('Использование: python client.py <путь_к_файлу.laz>')
        sys.exit(1)
    main(sys.argv[1])