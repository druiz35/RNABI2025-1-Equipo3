# Sistema Inteligente de Clasificación de Conducción Distractiva

## 📋 Descripción

Este proyecto implementa el **Módulo 2** del Sistema Inteligente Integrado para la empresa de transporte, enfocado en la clasificación de comportamientos distractivos en conductores mediante análisis de imágenes.

## 🎯 Funcionalidades

- **Clasificación de Imágenes**: Detecta 5 tipos de comportamientos de conducción
- **Interfaz Web Intuitiva**: Aplicación Streamlit con diseño moderno
- **Historial de Predicciones**: Almacena y visualiza todas las clasificaciones realizadas
- **Análisis Estadístico**: Gráficos y métricas de rendimiento
- **Imágenes de Ejemplo**: Pruebas con dataset predefinido

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd Trabajo3
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar archivos necesarios
Asegúrate de que el archivo del modelo esté disponible:
```
notebooks/modulo2/Resnet18.pth
```

## 🏃‍♂️ Ejecución

### Ejecutar la aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📱 Uso de la Aplicación

### 1. Pestaña "Clasificación"
- **Subir imagen**: Selecciona una imagen desde tu dispositivo
- **Imágenes de ejemplo**: Prueba con imágenes predefinidas del dataset
- **Resultados**: Visualiza la clasificación y nivel de confianza

### 2. Pestaña "Historial"
- Revisa todas las predicciones realizadas
- Visualiza imágenes analizadas anteriormente
- Estadísticas rápidas del uso

### 3. Pestaña "Estadísticas"
- Gráficos de distribución de comportamientos
- Análisis temporal de predicciones
- Métricas de rendimiento del modelo

### 4. Pestaña "Información"
- Documentación del sistema
- Descripción de comportamientos detectados
- Guía de uso

## 🔍 Comportamientos Detectados

| Clase | Descripción |
|-------|-------------|
| Conducción segura | Conductor atento y en postura correcta |
| Hablando por teléfono | Uso del teléfono para hablar |
| Enviando mensajes | Escribiendo o enviando mensajes |
| Girando | Realizando giro o cambio de dirección |
| Otras actividades | Actividades distractivas no categorizadas |

## 🤖 Modelo Utilizado

- **Arquitectura**: ResNet18
- **Entrenamiento**: Transfer Learning con ImageNet
- **Clases**: 5 comportamientos diferentes
- **Precisión**: Optimizada para detección de distracciones

## 📊 Métricas de Evaluación

- **Confianza**: Probabilidad de clasificación correcta
- **Precisión**: Exactitud en identificación de comportamientos
- **Recall**: Capacidad de detectar casos de distracción

## 🛠️ Estructura del Proyecto

```
Trabajo3/
├── app.py                 # Aplicación principal Streamlit
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
└── notebooks/
    └── modulo2/
        ├── m2_modelling.ipynb    # Notebook de entrenamiento
        ├── Resnet18.pth          # Modelo entrenado
        └── image_set_*.png       # Imágenes de ejemplo
```

## ⚠️ Requisitos del Sistema

- Python 3.8 o superior
- 4GB RAM mínimo (recomendado 8GB)
- Espacio en disco: 2GB para dependencias y modelo

## 🔧 Solución de Problemas

### Error: "No se pudo cargar el modelo"
- Verifica que el archivo `Resnet18.pth` esté en `notebooks/modulo2/`
- Asegúrate de que el archivo no esté corrupto

### Error: "ModuleNotFoundError"
- Instala las dependencias: `pip install -r requirements.txt`
- Verifica que estés en el entorno virtual correcto

### Error: "CUDA out of memory"
- El modelo está configurado para CPU por defecto
- Si tienes GPU, modifica `map_location=torch.device('cpu')` en `app.py`

## 📝 Notas Técnicas

- El modelo utiliza normalización ImageNet estándar
- Las imágenes se redimensionan a 224x224 píxeles
- El historial se almacena en la sesión de Streamlit (se pierde al cerrar)

## 🤝 Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Realiza los cambios
4. Envía un pull request

## 📄 Licencia

Este proyecto es parte del trabajo académico del curso de Redes Neuronales y Aprendizaje Profundo.

## 👥 Equipo

- Equipo 3 - RNABI2025-1
- Universidad Nacional de Colombia

---

**Nota**: Esta aplicación es parte del Módulo 2 del Sistema Inteligente Integrado para la empresa de transporte. 