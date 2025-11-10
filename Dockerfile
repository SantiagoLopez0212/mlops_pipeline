# Imagen base de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido del proyecto
COPY . .

# Exponer el puerto en el que correrá la API
EXPOSE 8000

# Comando para iniciar el servidor con FastAPI
CMD ["uvicorn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000"]

