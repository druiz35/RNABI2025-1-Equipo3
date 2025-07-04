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
from notebooks.modulo1.modulo1 import run_module1

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Clasificación de Conducción Distractiva",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración del sidebar
st.sidebar.title("MODULOS")
st.sidebar.markdown("### Módulo 1: Predicción de Demanda")
st.sidebar.markdown("### Módulo 2: Clasificación de Imágenes de Conducción")
st.sidebar.markdown("### Módulo 3: Sistema de Recomendación de Destinos de Viaje")

# Título principal
st.title("🛠️ Herramienta Web para el Análisis Inteligente de Transporte")

st.header("4.1 Introducción")
st.markdown("""
La presente herramienta web ha sido desarrollada como una solución integral para la gestión, análisis y toma de decisiones en el sector transporte, combinando técnicas avanzadas de inteligencia artificial y ciencia de datos. Su objetivo principal es facilitar a empresas, investigadores y tomadores de decisiones el acceso a modelos predictivos, clasificadores automáticos y sistemas de recomendación, todo desde una interfaz intuitiva y accesible desde cualquier navegador.

A través de esta plataforma, el usuario puede interactuar con distintos módulos inteligentes que permiten anticipar la demanda de rutas, analizar el comportamiento de conductores y recibir recomendaciones personalizadas de destinos, optimizando así la operación y la experiencia de los usuarios finales.

Esta herramienta es el resultado de un trabajo colaborativo y multidisciplinario, integrando conocimientos de análisis de datos, machine learning, desarrollo web y experiencia de usuario, con el fin de aportar valor real y tangible al sector transporte.
""")

st.header("4.2 Tecnologías utilizadas")
st.markdown("""
Esta herramienta se apoya en un stack tecnológico moderno y robusto, que garantiza tanto la eficiencia en el procesamiento como la facilidad de uso:

- **Python**: Lenguaje principal para el desarrollo de la lógica y los modelos.
- **Streamlit**: Framework para la creación de aplicaciones web interactivas de manera rápida y sencilla, ideal para prototipos y despliegue de soluciones de ciencia de datos.
- **PyTorch**: Biblioteca de deep learning utilizada para la construcción y entrenamiento de modelos de clasificación de imágenes.
- **scikit-learn**: Herramientas de machine learning para tareas de recomendación y análisis de datos.
- **Pandas y NumPy**: Manipulación y análisis eficiente de grandes volúmenes de datos.
- **Matplotlib y Seaborn**: Visualización avanzada de resultados, métricas y tendencias.
- **Otras librerías**: PIL para procesamiento de imágenes, y utilidades estándar de Python para manejo de archivos y fechas.

El uso de estas tecnologías permite integrar modelos complejos y visualizaciones ricas en una experiencia de usuario fluida y amigable. Además, la arquitectura modular facilita la escalabilidad y el mantenimiento del sistema, permitiendo la incorporación de nuevos módulos o funcionalidades en el futuro.
""")

st.header("4.3 Descripción de la interfaz")
st.markdown("""
La interfaz de la herramienta está diseñada bajo principios de simplicidad y claridad, permitiendo que cualquier usuario, sin importar su nivel técnico, pueda aprovechar al máximo las funcionalidades ofrecidas.

- **Estructura en bloques**: Cada módulo se presenta como un bloque independiente, con su propio formulario de entrada y visualización de resultados.
- **Inputs intuitivos**: Los parámetros requeridos por cada módulo (como número de días a predecir, imágenes a analizar o ID de usuario) se solicitan mediante controles sencillos como cajas numéricas, selectores y botones.
- **Visualización inmediata**: Los resultados, métricas y gráficas se muestran de forma clara y ordenada justo después de cada acción, permitiendo una interpretación rápida y efectiva.
- **Navegación vertical**: El usuario puede desplazarse fácilmente entre los diferentes módulos y secciones, accediendo a la información y funcionalidades de manera secuencial.
- **Mensajes y ayudas contextuales**: Se incluyen descripciones, recomendaciones y advertencias para guiar al usuario durante el uso de la herramienta.

**Ejemplo de uso:**
- Un planificador de rutas puede anticipar la demanda futura de los destinos turísticos más populares y ajustar la oferta de transporte en consecuencia.
- Un supervisor de seguridad vial puede analizar imágenes de conductores y detectar comportamientos distractivos de manera automática.
- Un turista o usuario final puede recibir recomendaciones personalizadas de destinos según su historial y preferencias, mejorando su experiencia de viaje.

Esta organización facilita la experimentación, el análisis comparativo y la toma de decisiones informadas en tiempo real, tanto para usuarios técnicos como no técnicos.
""")

st.header("4.4 Funcionalidades")
st.markdown("""
La herramienta integra tres módulos principales, cada uno orientado a resolver un problema específico dentro del ámbito del transporte inteligente:

- **Módulo 1: Predicción de demanda de rutas o destinos turísticos**
    - Permite anticipar la demanda futura en las rutas más populares, ayudando a planificar recursos y optimizar la operación.
    - El usuario selecciona el horizonte de predicción (número de días) y obtiene gráficas, métricas y análisis detallados para cada ruta.
    - Ideal para la gestión de flotas, planificación de servicios y análisis de tendencias turísticas.
    - **Beneficio:** Reduce la incertidumbre y mejora la asignación de recursos.

- **Módulo 2: Clasificación automática de imágenes de conducción**
    - Analiza imágenes de conductores para detectar comportamientos distractivos o inseguros mediante modelos de deep learning.
    - El usuario sube una imagen y recibe una clasificación automática, junto con métricas de confianza y recomendaciones.
    - Útil para empresas de transporte, aseguradoras y proyectos de seguridad vial.
    - **Beneficio:** Contribuye a la prevención de accidentes y mejora la seguridad en carretera.

- **Módulo 3: Recomendación personalizada de destinos o rutas**
    - Ofrece sugerencias de destinos turísticos o rutas a partir de las preferencias y el historial de cada usuario.
    - El usuario puede ingresar su ID o seleccionar su nombre para recibir recomendaciones personalizadas, visualizando además la popularidad y el tipo de cada destino sugerido.
    - Facilita la personalización de la experiencia y la promoción de destinos menos conocidos.
    - **Beneficio:** Aumenta la satisfacción del usuario y fomenta el descubrimiento de nuevas opciones.

Cada módulo puede ser utilizado de manera independiente, permitiendo adaptar la herramienta a diferentes necesidades y escenarios de uso. Además, la integración de los tres módulos en una sola plataforma potencia el análisis cruzado y la toma de decisiones estratégicas.
""")

st.markdown("---")

# ─── MÓDULO 1: Predicción de Demanda ────────────────────
st.header("📈 Módulo 1: Predicción de Demanda de Transporte")
st.markdown("Introduce sólo el horizonte (días) y ejecuta.")

horizon = st.number_input(
    "Horizonte (días)", min_value=1, max_value=365, value=30
)

if st.button("▶ Ejecutar Módulo 1"):
    with st.spinner("Generando predicciones…"):
        resultados = run_module1(horizon)

    for ruta, info in resultados.items():
        st.subheader(f"Ruta: {ruta}")
        st.subheader("📊 Métricas de Backtest")
        st.table(info["metrics"])
        st.subheader("📈 Demanda Sintética")
        st.pyplot(info["fig_demand"])
        st.subheader("🔍 Descomposición de la Serie")
        st.pyplot(info["fig_decomp"])

st.markdown("---")

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
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de predicciones", len(history_df))
        with col2:
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
                    st.markdown(f"**Fecha:** {row['timestamp']}")
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
        fig, ax = plt.subplots(figsize=(10, 6))
        behavior_counts.plot(kind='bar', ax=ax, color=['#4ecdc4', '#ff6b6b', '#45b7d1', '#96ceb4', '#feca57'])
        ax.set_title('Frecuencia de Comportamientos Detectados')
        ax.set_xlabel('Comportamiento')
        ax.set_ylabel('Frecuencia')
        plt.xticks(rotation=45, ha='right')
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

def update_destination_rep(recommendations_df):
    """
    Actualiza un archivo CSV que almacena el número de veces que cada destino
    ha sido recomendado.

    Args:
        recommendations_df (pd.DataFrame): DataFrame con al menos la columna 'DestinationID'.
        reporte_path (str): Ruta del archivo CSV donde se guardará el reporte acumulado.
    """
    reporte_path="./notebooks/modulo3/destinationRep.csv"

    # Verificar que exista la columna esperada
    if 'DestinationID' not in recommendations_df.columns:
        raise ValueError("El DataFrame de recomendaciones debe contener la columna 'DestinationID'.")

    # Leer el archivo existente o crear uno vacío si no existe
    if os.path.exists(reporte_path):
        reporte_df = pd.read_csv(reporte_path)
    else:
        # Crear DataFrame vacío con las columnas necesarias
        reporte_df = pd.DataFrame(columns=['DestinationID', 'Recomendaciones'])
    
    # Contar cuántas veces aparece cada DestinationID en este nuevo lote
    nuevas_recomendaciones = recommendations_df['DestinationID'].value_counts().reset_index()
    nuevas_recomendaciones.columns = ['DestinationID', 'Nuevas']

    # Combinar con el historial
    reporte_df = pd.merge(
        reporte_df,
        nuevas_recomendaciones,
        on='DestinationID',
        how='outer'
    )

    # Rellenar NaN con 0 y sumar
    reporte_df['Recomendaciones'] = reporte_df['Recomendaciones'].fillna(0) + reporte_df['Nuevas'].fillna(0)

    # Eliminar columna auxiliar
    reporte_df.drop(columns=['Nuevas'], inplace=True)

    # Guardar actualizado
    reporte_df.to_csv(reporte_path, index=False)

# Función para cargar el modelo
@st.cache_resource
def load_model3():
    return CollaborativeRecommendator()

# Cargar modelo
recommendator = load_model3()

# Carga la base de datos
m3_df = pd.read_csv(recommendator.MERGED_DF_PATH)

# Pestañas principales
tab_3_1, tab_3_2 = st.tabs(["👥 Recomendación por Nombre", "📖 Recomendación por ID"])

with tab_3_1:
    st.header("Recomendación por Nombre")

    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que el archivo recommendators.py esté disponible.")
        st.stop()

    st.write("Eligue entre algunos usuarios para ver sus recomendaciones personalizadas")

    # Tomar primeros 10 usuarios: UserID + Name_y
    primeros_usuarios = m3_df[['UserID', 'Name_y']].drop_duplicates().head(10)
    primeros_usuarios = primeros_usuarios[primeros_usuarios['UserID'] < 642]

    # Crear un diccionario: nombre => UserID
    usuarios_dict = dict(zip(primeros_usuarios['Name_y'], primeros_usuarios['UserID']))

    # ---------------------------
    # Selectbox con nombres
    # ---------------------------
    selected_name = st.selectbox(
        "Selecciona un usuario por nombre:",
        options=list(usuarios_dict.keys())
    )

    # ---------------------------
    # Botón para generar recomendación
    # ---------------------------
    if st.button("Generar recomendaciones"):
        # Mapear nombre a ID
        selected_user_id = usuarios_dict[selected_name]

        with st.spinner(f"Generando recomendaciones para **{selected_name}** (User ID: {selected_user_id})..."):
            recommendations = recommendator.recommend(int(selected_user_id))

            if recommendations.empty:
                st.warning("¡No se encontraron recomendaciones para este usuario!")
            else:
                st.success(f"¡Recomendaciones para **{selected_name}**!")
                st.dataframe(recommendations)
                update_destination_rep(recommendations)
                # ---------------------------
                # Nota opcional
                # ---------------------------
                st.caption("Dataset: India Travel Recommender | Modelo colaborativo | Desarrollado para proyectos educativos.")

with tab_3_2:
    st.header("Recomendación por ID")

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
                update_destination_rep(recommendations)
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