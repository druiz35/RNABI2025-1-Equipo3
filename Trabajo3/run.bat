@echo off
REM Script de inicio rápido para el Sistema de Clasificación de Conducción Distractiva

echo 🚗 Iniciando Sistema de Clasificación de Conducción Distractiva
echo ================================================================

REM Verificar si existe el entorno virtual
if not exist "venv" (
    echo 🔧 Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Error al crear el entorno virtual
        pause
        exit /b 1
    )
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias si no están instaladas
if not exist "venv\Lib\site-packages\streamlit" (
    echo 📦 Instalando dependencias...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Error al instalar dependencias
        pause
        exit /b 1
    )
)

REM Verificar que el modelo existe
if not exist "notebooks\modulo2\Resnet18.pth" (
    echo ⚠️  Advertencia: No se encontró el modelo Resnet18.pth
    echo    Asegúrate de tener el archivo del modelo antes de continuar
)

REM Ejecutar pruebas
echo 🧪 Ejecutando pruebas...
python test_app.py

REM Iniciar la aplicación
echo 🚀 Iniciando aplicación Streamlit...
echo    La aplicación se abrirá en: http://localhost:8501
echo    Presiona Ctrl+C para detener
echo.

streamlit run app.py

pause 