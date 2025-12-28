"""
Модуль для веб-деплоя приложения
"""

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard import app
from src.data_loader import generate_sample_data

# Создаем сервер для Gunicorn
server = app.server

if __name__ == '__main__':
    # Запуск в режиме разработки
    app.run_server(debug=True, port=8050)