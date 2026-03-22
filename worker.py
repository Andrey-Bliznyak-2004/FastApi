from tasks import app

if __name__ == '__main__':
    # Добавлен параметр '--pool=solo' для корректной работы на Windows
    app.worker_main(['worker', '--loglevel=info', '--concurrency=1', '--pool=solo'])