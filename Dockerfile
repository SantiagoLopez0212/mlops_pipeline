# Imagen base de Python (estable y ligera)
FROM python:3.11-slim

# Evita que Python genere archivos .pyc y usa salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivo de dependencias primero (para usar cache de Docker)
COPY requirements.txt .

# Instalar dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto al contenedor
COPY . .

# Exponer el puerto donde correrá FastAPI
EXPOSE 8000

# Comando para iniciar el servidor con Uvicorn
CMD ["uvicorn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
