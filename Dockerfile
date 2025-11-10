# Imagen base
FROM python:3.12-slim

# Evitar buffering
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto al contenedor
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Ejecutar el servicio de la API
CMD ["uvicorn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
