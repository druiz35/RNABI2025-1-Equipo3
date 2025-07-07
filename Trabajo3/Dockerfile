# Usar una imagen base más ligera con Python
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo los archivos de requirements primero para aprovechar cache de Docker
COPY requirements-deployment.txt .

# Instalar dependencias de Python con optimizaciones para espacio
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements-deployment.txt

# Copiar el resto del código
COPY app.py .
COPY notebooks/modulo2/classifiers.py ./notebooks/modulo2/
COPY notebooks/modulo2/Resnet18.pth ./notebooks/modulo2/
COPY notebooks/modulo2/*.png ./notebooks/modulo2/

# Crear directorio para imágenes de prueba
RUN mkdir -p ./notebooks/modulo2/test

# Exponer puerto
EXPOSE 8501

# Comando de salud para verificar el contenedor
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 