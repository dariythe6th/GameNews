# Dockerfile
# Используем Python-образ, оптимизированный для маленьких размеров и быстрой сборки.
FROM python:3.12-slim

# Установим системные зависимости, необходимые для psycopg2-binary/asyncpg и scikit-learn
# (например, для компиляции numpy/scipy и клиента PostgreSQL)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY main.py .

# Expose the port uvicorn runs on
EXPOSE 8080

# Команда запуска приложения
# Используем `python main.py`, который внутри себя вызывает uvicorn.run("main:app", ...)
CMD ["python", "main.py"]