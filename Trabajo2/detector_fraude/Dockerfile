# Usar una imagen base más ligera
FROM python:3.10-slim

# Variables de entorno para optimizar Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar solo las dependencias necesarias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear y establecer el directorio de trabajo
WORKDIR /app

# Copiar solo los archivos necesarios primero (para aprovechar la caché de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Recolectar archivos estáticos (si es necesario)
RUN python manage.py collectstatic --noinput || true

# Exponer el puerto
EXPOSE 8000

# Usar gunicorn con workers optimizados
CMD ["gunicorn", "detector_fraude.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", "--timeout", "60"] 