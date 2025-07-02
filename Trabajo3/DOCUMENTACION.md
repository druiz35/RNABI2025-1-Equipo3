# Documentación Técnica - Módulo 2: Clasificación de Conducción Distractiva

## 📋 Resumen Ejecutivo

El Módulo 2 implementa un sistema de clasificación de imágenes para detectar comportamientos distractivos en conductores, utilizando técnicas de aprendizaje profundo con la arquitectura ResNet18. Este sistema contribuye significativamente a mejorar la seguridad vial en la empresa de transporte.

## 🎯 Objetivos Cumplidos

### ✅ Objetivos Específicos Alcanzados

1. **Modelo de Clasificación Entrenado**: Implementación exitosa de ResNet18 con transfer learning
2. **Detección de Comportamientos**: Identificación de 5 tipos de comportamientos de conducción
3. **Interfaz Web Integrada**: Aplicación Streamlit completa y funcional
4. **Métricas de Evaluación**: Sistema de tracking de precisión y confianza
5. **Análisis de Distracciones**: Identificación de patrones de comportamiento distractivo

## 🤖 Arquitectura del Modelo

### ResNet18 - Transfer Learning

```python
# Configuración del modelo
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 clases de comportamiento
```

**Características técnicas:**
- **Arquitectura base**: ResNet18 pre-entrenada en ImageNet
- **Capas de salida**: 5 neuronas (una por clase de comportamiento)
- **Optimizador**: Adam con learning rate 1e-4
- **Función de pérdida**: CrossEntropyLoss
- **Normalización**: ImageNet estándar (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Preprocesamiento de Imágenes

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Redimensionar a 224x224
    transforms.ToTensor(),                   # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # Normalización ImageNet
])
```

## 📊 Clases de Comportamiento

| Índice | Clase (Inglés) | Clase (Español) | Descripción |
|--------|----------------|-----------------|-------------|
| 0 | other_activities | Otras actividades | Actividades distractivas no categorizadas |
| 1 | safe_driving | Conducción segura | Conductor atento y en postura correcta |
| 2 | talking_phone | Hablando por teléfono | Uso del teléfono para hablar |
| 3 | texting_phone | Enviando mensajes | Escribiendo o enviando mensajes de texto |
| 4 | turning | Girando | Realizando giro o cambio de dirección |

## 🏗️ Estructura de la Aplicación

### Arquitectura de la Interfaz Web

```
app.py
├── Configuración de página
├── Carga del modelo (@st.cache_resource)
├── Funciones de preprocesamiento
├── Funciones de predicción
├── Funciones de visualización
└── Pestañas principales:
    ├── 📸 Clasificación
    ├── 📊 Historial
    ├── 📈 Estadísticas
    └── ℹ️ Información
```

### Flujo de Datos

1. **Entrada**: Imagen del conductor (upload o ejemplo)
2. **Preprocesamiento**: Redimensionar, normalizar, convertir a tensor
3. **Predicción**: Modelo ResNet18 → probabilidades por clase
4. **Post-procesamiento**: Softmax → clase predicha + confianza
5. **Visualización**: Gráficos de barras, métricas, historial
6. **Almacenamiento**: Guardar en session_state para historial

## 📈 Métricas de Rendimiento

### Métricas Implementadas

- **Confianza**: Probabilidad de que la clasificación sea correcta
- **Precisión**: Exactitud en la identificación de comportamientos
- **Distribución de clases**: Frecuencia de cada comportamiento detectado
- **Análisis temporal**: Predicciones por hora del día
- **Tendencias**: Evolución de la confianza promedio

### Visualizaciones Generadas

1. **Gráfico de barras**: Probabilidades por clase de comportamiento
2. **Histograma**: Distribución de niveles de confianza
3. **Gráfico de líneas**: Predicciones por hora del día
4. **Métricas en tiempo real**: Contadores y promedios

## 🔧 Funcionalidades Implementadas

### 1. Pestaña de Clasificación

**Características:**
- Subida de imágenes desde dispositivo
- Imágenes de ejemplo predefinidas
- Previsualización de imagen subida
- Clasificación en tiempo real
- Visualización de resultados con gráficos

**Flujo de trabajo:**
```python
# 1. Cargar imagen
image = Image.open(uploaded_file)

# 2. Preprocesar
image_tensor = preprocess_image(image)

# 3. Predecir
predicted_class, confidence, probabilities = predict_image(model, image_tensor)

# 4. Visualizar
plot_predictions(probabilities, CLASSES_ES)
```

### 2. Pestaña de Historial

**Características:**
- Almacenamiento de todas las predicciones
- Visualización de imágenes analizadas
- Métricas rápidas (total, confianza promedio, conducción segura)
- Expansores para cada predicción
- Botón para limpiar historial

**Almacenamiento:**
```python
# Convertir imagen a base64 para almacenamiento
img_buffer = io.BytesIO()
image.save(img_buffer, format='PNG')
img_str = base64.b64encode(img_buffer.getvalue()).decode()

# Guardar en session_state
st.session_state.prediction_history.append({
    'timestamp': timestamp,
    'image': img_str,
    'prediction': prediction,
    'confidence': confidence
})
```

### 3. Pestaña de Estadísticas

**Análisis implementados:**
- Distribución de comportamientos detectados
- Histograma de niveles de confianza
- Métricas de rendimiento (mín, máx, promedio)
- Análisis temporal por hora del día

**Gráficos generados:**
```python
# Gráfico de barras para comportamientos
behavior_counts.plot(kind='bar', ax=ax, color=['#4ecdc4', '#ff6b6b', '#45b7d1', '#96ceb4', '#feca57'])

# Histograma de confianza
ax.hist(history_df['confidence'], bins=20, color='#4ecdc4', alpha=0.7)

# Gráfico temporal
hourly_counts.plot(kind='line', marker='o', ax=ax, color='#45b7d1')
```

### 4. Pestaña de Información

**Contenido documentado:**
- Objetivo del módulo
- Descripción de comportamientos detectados
- Información técnica del modelo
- Guía de uso paso a paso
- Limitaciones del sistema

## 🚀 Instalación y Configuración

### Requisitos del Sistema

- **Python**: 3.8 o superior
- **RAM**: 4GB mínimo (8GB recomendado)
- **Espacio en disco**: 2GB para dependencias y modelo
- **Sistema operativo**: Windows, macOS, Linux

### Dependencias Principales

```txt
streamlit>=1.28.0      # Interfaz web
torch>=2.0.0          # Framework de deep learning
torchvision>=0.15.0   # Transformaciones de imágenes
Pillow>=9.0.0         # Procesamiento de imágenes
matplotlib>=3.5.0     # Visualizaciones
pandas>=1.3.0         # Análisis de datos
```

### Scripts de Automatización

1. **setup.py**: Configuración automática del entorno
2. **run.sh** (Linux/Mac): Inicio rápido con verificación
3. **run.bat** (Windows): Inicio rápido con verificación
4. **test_app.py**: Pruebas automatizadas del sistema

## 📊 Resultados y Evaluación

### Métricas de Rendimiento Esperadas

- **Precisión general**: >85% en dataset de prueba
- **Confianza promedio**: >80% para predicciones correctas
- **Tiempo de inferencia**: <2 segundos por imagen
- **Uso de memoria**: <2GB RAM durante operación

### Casos de Uso Validados

1. **Conducción segura**: Alta precisión en detección
2. **Uso de teléfono**: Detección efectiva de distracciones
3. **Envío de mensajes**: Identificación de comportamiento de riesgo
4. **Giros**: Detección de cambios de dirección
5. **Otras actividades**: Clasificación de comportamientos no categorizados

## 🔍 Análisis de Limitaciones

### Limitaciones Identificadas

1. **Calidad de imagen**: Mejor rendimiento con imágenes claras y bien iluminadas
2. **Ángulo de captura**: El modelo funciona mejor con vistas frontales del conductor
3. **Resolución**: Imágenes de baja resolución pueden afectar la precisión
4. **Variabilidad**: Diferentes tipos de vehículos pueden requerir ajustes

### Mejoras Futuras

1. **Data augmentation**: Más transformaciones para robustez
2. **Ensemble models**: Combinación de múltiples arquitecturas
3. **Real-time processing**: Procesamiento en tiempo real con video
4. **Mobile deployment**: Optimización para dispositivos móviles

## 🛡️ Consideraciones de Seguridad

### Privacidad de Datos

- **Almacenamiento local**: Las imágenes se procesan localmente
- **Sin persistencia**: El historial se borra al cerrar la aplicación
- **Sin tracking**: No se envían datos a servidores externos

### Validación de Entrada

- **Tipos de archivo**: Solo PNG, JPG, JPEG
- **Tamaño de imagen**: Limitado por memoria disponible
- **Sanitización**: Validación de formato de imagen

## 📝 Conclusiones

### Logros Principales

1. ✅ **Sistema funcional**: Aplicación completa y operativa
2. ✅ **Interfaz intuitiva**: Diseño moderno y fácil de usar
3. ✅ **Análisis robusto**: Métricas y visualizaciones comprehensivas
4. ✅ **Documentación completa**: Guías y documentación técnica
5. ✅ **Automatización**: Scripts de instalación y configuración

### Impacto en la Empresa

- **Seguridad vial mejorada**: Detección temprana de distracciones
- **Monitoreo eficiente**: Sistema automatizado de clasificación
- **Análisis de patrones**: Identificación de comportamientos de riesgo
- **Cumplimiento normativo**: Herramienta para auditorías de seguridad

### Próximos Pasos

1. **Integración con módulos 1 y 3**: Sistema completo de la empresa
2. **Despliegue en producción**: Implementación en servidores de la empresa
3. **Entrenamiento continuo**: Mejora del modelo con nuevos datos
4. **Expansión de clases**: Detección de más tipos de distracciones

---

**Documento generado**: $(date)
**Versión**: 1.0
**Equipo**: RNABI2025-1-Equipo3
**Módulo**: 2 - Clasificación de Conducción Distractiva 