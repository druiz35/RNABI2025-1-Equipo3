#!/bin/bash

# Script de inicio rápido para el Sistema de Clasificación de Conducción Distractiva

echo "🚗 Iniciando Sistema de Clasificación de Conducción Distractiva"
echo "================================================================"

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "🔧 Creando entorno virtual..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error al crear el entorno virtual"
        exit 1
    fi
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias si no están instaladas
if [ ! -f "venv/lib/python*/site-packages/streamlit" ]; then
    echo "📦 Instalando dependencias..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error al instalar dependencias"
        exit 1
    fi
fi

# Verificar que el modelo existe
if [ ! -f "notebooks/modulo2/Resnet18.pth" ]; then
    echo "⚠️  Advertencia: No se encontró el modelo Resnet18.pth"
    echo "   Asegúrate de tener el archivo del modelo antes de continuar"
fi

# Ejecutar pruebas
echo "🧪 Ejecutando pruebas..."
python test_app.py

# Iniciar la aplicación
echo "🚀 Iniciando aplicación Streamlit..."
echo "   La aplicación se abrirá en: http://localhost:8501"
echo "   Presiona Ctrl+C para detener"
echo ""

streamlit run app.py 