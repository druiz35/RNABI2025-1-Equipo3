import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import os
import io
import base64
from notebooks.modulo2.classifiers import DriverClassifier
from notebooks.modulo3.recommendators import CollaborativeRecommendator

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Clasificación de Conducción Distractiva",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración del sidebar
st.sidebar.title("Configuración")
st.sidebar.markdown("### Módulo 2: Clasificación de Imágenes")
st.sidebar.markdown("### Módulo 3: Recomendación de Destinos de Viaje")

# Título principal
st.title("🚗 Sistema Inteligente de Clasificación de Conducción Distractiva")
st.markdown("---")

# Clases de comportamiento
CLASSES = [
    "other_actvities",
    "safe_driving", 
    "talking_phone",
    "texting_phone",
    "turning"
]

CLASSES_ES = [
    "Otras actividades",
    "Conducción segura",
    "Hablando por teléfono", 
    "Enviando mensajes",
    "Girando"
]

# Función para cargar el modelo
@st.cache_resource
def load_model():
    return DriverClassifier()

# Función para preprocesar imagen
def preprocess_image(image):
    """Preprocesa la imagen para el modelo"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Aplicar transformaciones
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# Función para hacer predicción
def predict_image(model, image):
    # Guardar la imagen temporalmente para usar el método predict de la clase
    temp_path = "temp_image.png"
    image.save(temp_path)
    pred = model.predict(temp_path)
    os.remove(temp_path)
    # No tenemos probabilidades, solo la clase predicha
    return pred

# Función para crear gráfico de barras
def plot_predictions(probabilities, classes_es):
    """Crea un gráfico de barras con las probabilidades"""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes_es, probabilities, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
    
    # Personalizar gráfico
    ax.set_title('Probabilidades de Clasificación', fontsize=16, fontweight='bold')
    ax.set_xlabel('Comportamientos', fontsize=12)
    ax.set_ylabel('Probabilidad', fontsize=12)
    ax.set_ylim(0, 1)
    
    # Rotar etiquetas
    plt.xticks(rotation=45, ha='right')
    
    # Agregar valores en las barras
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Función para guardar historial
def save_prediction_history(image, prediction, confidence, timestamp):
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    st.session_state.prediction_history.append({
        'timestamp': timestamp,
        'image': img_str,
        'prediction': prediction,
        'confidence': confidence
    })

# Cargar modelo
model = load_model()

if model is None:
    st.error("No se pudo cargar el modelo. Verifica que el archivo Resnet18.pth esté disponible.")
    st.stop()

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["📸 Clasificación", "📊 Historial", "📈 Estadísticas", "ℹ️ Información"])

with tab1:
    st.header("Clasificación de Imágenes de Conducción")
    
    # --- Estado para limpiar resultado al cambiar imagen ---
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'predicted_class' not in st.session_state:
        st.session_state.predicted_class = None

    uploaded_file = st.file_uploader(
        "Sube una imagen del conductor:",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )

    # Limpiar resultado si cambia la imagen
    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.predicted_class = None
            st.session_state.last_uploaded_file = uploaded_file.name

        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_container_width=True)
        if st.button("🔍 Clasificar Imagen", type="primary"):
            temp_path = "temp_image.png"
            image.save(temp_path)
            predicted_class = model.predict(temp_path)
            st.session_state.predicted_class = predicted_class
            if predicted_class in CLASSES:
                idx = CLASSES.index(predicted_class)
                clase_es = CLASSES_ES[idx]
            else:
                clase_es = predicted_class
            st.markdown(f"**Comportamiento detectado:** {clase_es}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_prediction_history(image, clase_es, 1.0, timestamp)
            st.success("✅ Clasificación completada y guardada en el historial")
        elif st.session_state.predicted_class is not None:
            # Mostrar el resultado de la última predicción si existe
            predicted_class = st.session_state.predicted_class
            if predicted_class in CLASSES:
                idx = CLASSES.index(predicted_class)
                clase_es = CLASSES_ES[idx]
            else:
                clase_es = predicted_class
            st.markdown(f"**Comportamiento detectado:** {clase_es}")
    else:
        st.session_state.last_uploaded_file = None
        st.session_state.predicted_class = None

with tab2:
    st.header("📊 Historial de Predicciones")
    
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        # Crear DataFrame del historial
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Mostrar estadísticas rápidas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de predicciones", len(history_df))
        with col2:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Confianza promedio", f"{avg_confidence:.2%}")
        with col3:
            safe_driving_count = len(history_df[history_df['prediction'] == 'Conducción segura'])
            st.metric("Conducción segura", safe_driving_count)
        
        # Mostrar historial en tabla
        st.subheader("Historial Detallado")
        
        # Crear tabla con imágenes
        for i, row in history_df.iterrows():
            with st.expander(f"Predicción {i+1} - {row['timestamp']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Decodificar imagen
                    img_data = base64.b64decode(row['image'])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, caption="Imagen analizada", width=200)
                
                with col2:
                    st.markdown(f"**Comportamiento:** {row['prediction']}")
                    st.markdown(f"**Confianza:** {row['confidence']:.2%}")
                    st.markdown(f"**Fecha:** {row['timestamp']}")
                    
                    # Indicador de confianza
                    st.progress(row['confidence'])
        
        # Botón para limpiar historial
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.prediction_history = []
            st.rerun()
    
    else:
        st.info("No hay predicciones en el historial. ¡Sube una imagen para comenzar!")

with tab3:
    st.header("📈 Estadísticas y Análisis")
    
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Análisis de comportamientos
        st.subheader("Distribución de Comportamientos")
        
        behavior_counts = history_df['prediction'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            behavior_counts.plot(kind='bar', ax=ax, color=['#4ecdc4', '#ff6b6b', '#45b7d1', '#96ceb4', '#feca57'])
            ax.set_title('Frecuencia de Comportamientos Detectados')
            ax.set_xlabel('Comportamiento')
            ax.set_ylabel('Frecuencia')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Gráfico de confianza
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(history_df['confidence'], bins=20, color='#4ecdc4', alpha=0.7)
            ax.set_title('Distribución de Confianza')
            ax.set_xlabel('Confianza')
            ax.set_ylabel('Frecuencia')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Métricas adicionales
        st.subheader("Métricas de Rendimiento")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de predicciones", len(history_df))
        
        with col2:
            avg_conf = history_df['confidence'].mean()
            st.metric("Confianza promedio", f"{avg_conf:.2%}")
        
        with col3:
            max_conf = history_df['confidence'].max()
            st.metric("Confianza máxima", f"{max_conf:.2%}")
        
        with col4:
            min_conf = history_df['confidence'].min()
            st.metric("Confianza mínima", f"{min_conf:.2%}")
        
        # Análisis temporal
        st.subheader("Análisis Temporal")
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['hour'] = history_df['timestamp'].dt.hour
        
        fig, ax = plt.subplots(figsize=(12, 6))
        hourly_counts = history_df['hour'].value_counts().sort_index()
        hourly_counts.plot(kind='line', marker='o', ax=ax, color='#45b7d1')
        ax.set_title('Predicciones por Hora del Día')
        ax.set_xlabel('Hora')
        ax.set_ylabel('Número de Predicciones')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    else:
        st.info("No hay datos suficientes para mostrar estadísticas. ¡Realiza algunas predicciones primero!")

with tab4:
    st.header("ℹ️ Información del Sistema")
    
    st.subheader("🎯 Objetivo del Módulo 2")
    st.markdown("""
    Este módulo implementa un sistema de clasificación de imágenes para detectar comportamientos distractivos 
    en conductores, contribuyendo a mejorar la seguridad vial en la empresa de transporte.
    """)
    
    st.subheader("🔍 Comportamientos Detectados")
    
    behaviors_info = {
        "Conducción segura": "El conductor mantiene una postura adecuada y atención en la carretera",
        "Hablando por teléfono": "El conductor está usando el teléfono para hablar",
        "Enviando mensajes": "El conductor está escribiendo o enviando mensajes de texto",
        "Girando": "El conductor está realizando un giro o cambio de dirección",
        "Otras actividades": "Otras actividades distractivas no categorizadas"
    }
    
    for behavior, description in behaviors_info.items():
        with st.expander(f"📱 {behavior}"):
            st.write(description)
    
    st.subheader("🤖 Modelo Utilizado")
    st.markdown("""
    - **Arquitectura:** ResNet18
    - **Entrenamiento:** Transfer Learning con pesos pre-entrenados en ImageNet
    - **Clases:** 5 comportamientos diferentes
    - **Precisión:** Optimizada para detección de distracciones
    """)
    
    st.subheader("📊 Métricas de Evaluación")
    st.markdown("""
    El modelo evalúa cada predicción con:
    - **Confianza:** Probabilidad de que la clasificación sea correcta
    - **Precisión:** Exactitud en la identificación de comportamientos
    - **Recall:** Capacidad de detectar todos los casos de distracción
    """)
    
    st.subheader("🚀 Cómo Usar")
    st.markdown("""
    1. **Subir imagen:** Selecciona una imagen del conductor desde tu dispositivo
    2. **Clasificar:** Haz clic en el botón de clasificación
    3. **Revisar resultados:** Analiza la predicción y la confianza
    4. **Consultar historial:** Revisa todas las predicciones anteriores
    5. **Ver estadísticas:** Analiza tendencias y patrones
    """)
    
    st.subheader("⚠️ Limitaciones")
    st.markdown("""
    - La calidad de la imagen afecta la precisión de la clasificación
    - El modelo funciona mejor con imágenes claras y bien iluminadas
    - Se recomienda usar imágenes donde el conductor sea claramente visible
    """)


st.markdown("---")
st.markdown("---")
# Título principal
st.title("🌄 Sistema de Recomendación de Destinos de Viaje")
st.markdown("---")

# Argumentos
ARGSS3 = [
    "DestinationID",
    "Name",
    "State",
    "Type",
    "Popularity",
    "BestTimeToVisit"
]


ARGS3_ES = [
    "ID del destino",
    "Nombre",
    "Estado/Provincia",
    "Tipo",
    "Popularidad",
    "Mejor época para visitar"
]

# Función para cargar el modelo
@st.cache_resource
def load_model3():
    return CollaborativeRecommendator()

# Cargar modelo
recommendator = load_model3()

if model is None:
    st.error("No se pudo cargar el modelo. Verifica que el archivo recommendators.py esté disponible.")
    st.stop()

st.write("Introduce tu **User ID** para recibir recomendaciones personalizadas de destinos en India.")

# ---------------------------
# Inputs de usuario
# ---------------------------
user_id = st.number_input("User ID", min_value=1, step=1)

# ---------------------------
# Ejecutar recomendación
# ---------------------------
if st.button("Obtener recomendaciones"):
    with st.spinner("Generando recomendaciones..."):
        recommendations = recommendator.recommend(user_id)

        if recommendations.empty:
            st.warning("No se encontraron recomendaciones para este usuario.")
        else:
            st.success("¡Aquí están tus recomendaciones!")
            st.dataframe(recommendations)
            # ---------------------------
            # Nota opcional
            # ---------------------------
            st.caption("Dataset: India Travel Recommender | Modelo colaborativo | Desarrollado para proyectos educativos.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en la Empresa de Transporte</p>
        <p>Módulo 2: Clasificación de Conducción Distractiva</p>
    </div>
    """,
    unsafe_allow_html=True
) 