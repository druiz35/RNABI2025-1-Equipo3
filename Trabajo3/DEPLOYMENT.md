# 🚀 Guía de Deployment Optimizado

## Problema Identificado

El error `no space left on device` se debe a que PyTorch (`torch`) es una librería muy pesada, especialmente el archivo `libtorch_cpu.so` que puede ocupar varios GB durante el proceso de build.

## ✅ Soluciones Implementadas

### 1. **Requirements Optimizado**
- `requirements-deployment.txt`: Versión ligera solo para producción
- Usa versiones CPU-only de PyTorch: `torch==2.1.0+cpu`
- Elimina dependencias innecesarias

### 2. **Dockerfile Optimizado**
- Imagen base `python:3.9-slim` (más ligera)
- Multi-stage build para reducir tamaño final
- Cache de dependencias optimizado
- Flag `--no-cache-dir` para pip

### 3. **.dockerignore Configurado**
- Excluye archivos grandes innecesarios
- No incluye notebooks pesados ni datasets
- Mantiene solo archivos esenciales

## 🔧 Pasos para Deployment

### Opción A: Deployment Rápido
```bash
# 1. Usar script automatizado
./deploy.sh

# 2. Ir al directorio optimizado
cd deploy_temp

# 3. Deployar con el requirements optimizado
# Renombrar requirements-deployment.txt a requirements.txt
mv requirements-deployment.txt requirements.txt
```

### Opción B: Deployment Manual
```bash
# 1. Crear directorio limpio
mkdir deployment
cd deployment

# 2. Copiar solo archivos esenciales
cp ../app.py .
cp ../requirements-deployment.txt requirements.txt
cp ../Dockerfile .
cp -r ../notebooks/modulo2/ notebooks/

# 3. Limpiar archivos grandes
rm -f notebooks/modulo2/*.ipynb
rm -f notebooks/modulo2/image_set_*.png
```

## 📊 Optimizaciones Adicionales

### 1. **Reducir Tamaño del Modelo**
Si el modelo sigue siendo muy grande, considera:

```python
# Cuantización del modelo (en un script separado)
import torch

# Cargar modelo
model = torch.load('Resnet18.pth', map_location='cpu')

# Cuantizar a int8
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Guardar modelo cuantizado
torch.save(model_quantized.state_dict(), 'Resnet18_quantized.pth')
```

### 2. **Variables de Entorno para DigitalOcean**
```bash
# Limitar uso de memoria durante build
export PYTORCH_BUILD_NUMBER=0
export MAX_JOBS=1
```

### 3. **Alternativa: Usar CPU-only desde el inicio**
```txt
# requirements-minimal.txt
streamlit==1.28.1
torch==2.1.0+cpu --find-links https://download.pytorch.org/whl/torch_stable.html
torchvision==0.16.0+cpu --find-links https://download.pytorch.org/whl/torch_stable.html
Pillow==9.5.0
numpy==1.24.4
```

## 🆘 Soluciones de Emergencia

### Si el problema persiste:

1. **Aumentar el plan de DigitalOcean temporalmente**
   - Upgrade a un droplet con más espacio
   - Hacer el deployment
   - Downgrade después del deployment exitoso

2. **Usar un modelo pre-compilado más pequeño**
   - Buscar versiones optimizadas del modelo
   - Usar TorchScript para optimizar

3. **Deployment en dos fases**
   ```bash
   # Fase 1: Solo dependencias
   pip install streamlit pandas numpy Pillow
   
   # Fase 2: PyTorch optimizado
   pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

## 📈 Verificación del Deployment

1. **Comprobar tamaño del contenedor**
   ```bash
   docker images | grep your-app-name
   ```

2. **Verificar funcionamiento**
   ```bash
   curl http://localhost:8501/_stcore/health
   ```

3. **Monitorear uso de recursos**
   ```bash
   docker stats your-container-name
   ```

## 🎯 Resultados Esperados

Con estas optimizaciones deberías conseguir:
- ⬇️ **Reducción del 60-70%** en tamaño del contenedor
- ⚡ **Build 3x más rápido**
- 💾 **Menor uso de espacio en disco**
- ✅ **Deployment exitoso en DigitalOcean**

---

**💡 Tip**: Si nada funciona, considera usar un servicio como Hugging Face Spaces o Streamlit Cloud que están optimizados para aplicaciones ML. 