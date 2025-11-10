# ===== Imagen base ligera de Python =====
FROM python:3.11-slim

# ===== Variables de entorno =====
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ===== Directorio de trabajo dentro del contenedor =====
WORKDIR /app

# ===== Copiar dependencias primero (para cache de Docker) =====
COPY requirements.txt .

# ===== Instalar dependencias =====
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== Copiar todo el proyecto =====
COPY . .

# ===== Exponer puerto para FastAPI =====
EXPOSE 8000

# ===== Comando para iniciar el servidor =====
CMD ["uvicorn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
