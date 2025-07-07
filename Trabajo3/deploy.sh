#!/bin/bash

echo "🚀 Preparando deployment optimizado para DigitalOcean..."

# Verificar que estamos en el directorio correcto
if [ ! -f "app.py" ]; then
    echo "❌ Error: No se encuentra app.py. Asegúrate de estar en el directorio correcto."
    exit 1
fi

# Crear directorio temporal para deployment
echo "📁 Creando estructura de deployment..."
mkdir -p deploy_temp
cd deploy_temp

# Copiar solo archivos esenciales
echo "📋 Copiando archivos esenciales..."
cp ../app.py .
cp ../requirements-deployment.txt .
cp ../Dockerfile .
cp ../.dockerignore .

# Crear estructura de directorios necesaria
mkdir -p notebooks/modulo2

# Copiar solo el clasificador y modelo
cp ../notebooks/modulo2/classifiers.py notebooks/modulo2/
cp ../notebooks/modulo2/Resnet18.pth notebooks/modulo2/

# Copiar solo las imágenes de prueba necesarias (las más pequeñas)
cp ../notebooks/modulo2/safe_test.png notebooks/modulo2/ 2>/dev/null || echo "⚠️  safe_test.png no encontrado"
cp ../notebooks/modulo2/phone_test.png notebooks/modulo2/ 2>/dev/null || echo "⚠️  phone_test.png no encontrado"

# Verificar tamaño del modelo
model_size=$(du -m notebooks/modulo2/Resnet18.pth | cut -f1)
echo "📊 Tamaño del modelo: ${model_size}MB"

if [ $model_size -gt 100 ]; then
    echo "⚠️  Advertencia: El modelo es grande (${model_size}MB). Considera usar cuantización."
fi

# Mostrar estructura final
echo "📂 Estructura de deployment:"
find . -name "*.py" -o -name "*.pth" -o -name "*.txt" -o -name "Dockerfile" | sort

echo ""
echo "✅ Deployment preparado en ./deploy_temp/"
echo ""
echo "🔧 Pasos siguientes para DigitalOcean:"
echo "1. cd deploy_temp"
echo "2. git init && git add . && git commit -m 'Deploy optimized app'"
echo "3. git remote add origin <tu-repo-url>"
echo "4. git push origin main"
echo ""
echo "💡 Alternativa: usar requirements-deployment.txt en lugar de requirements.txt" 