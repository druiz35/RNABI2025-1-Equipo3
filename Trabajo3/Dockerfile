# Usar imagen base optimizada
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .

# Instalar PyTorch CPU primero desde el índice específico
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu torchvision==0.16.0+cpu

# Instalar dependencias compatibles SIN especificar versiones exactas para evitar conflictos
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    Pillow \
    plotly \
    matplotlib \
    seaborn \
    scikit-learn \
    statsmodels \
    "tensorflow>=2.10,<2.14"

# Copiar toda la aplicación
COPY . .

# Exponer puerto
EXPOSE 8501

# Comando de salud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Ejecutar aplicación
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 