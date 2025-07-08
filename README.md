<h1>
  Equipo 3
  <br>Redes Neuronales y Algoritmos Bioinspirados
  <br>2025-1
</h1>
<h3>
  Profesor: Juan David Ospina Arango
  <br>Monitor: Andrés Mauricio Zapata Rincón
  <br>Universidad Nacional de Colombia sede Medellín
</h3>

<h1>Descripción</h1>
<p>Repositorio que recopila el código para los entregables desarrollados para la asignatura Redes Neuronales y Algoritmos Bioinspirados del semestre 2025-1 de la Universidad Nacional de Colombia sede Medellín.</p>
<p>Cada entregable se encuentra dividido en carpetas que contienen todo el código y demás contenido desarrollado para el entregable.</p>

<h1>Expectativas de aprendizaje de cada trabajo</h1>
<h3>Trabajo 1: Optimización heurística</h3>
<ul>
  <li>Entender cómo cada uno de estos métodos de optimización bioinspirados podrían encajar en distintos problemas.</li>
  <li>Implementar en el lenguaje R los siguientes métodos de optimización: Descenso de gradiente, Algoritmos evolutivos, Colonias de hormigas, Enjambre de partículas, Evolución diferencial.</li>
  <li>Comparar los resultados de cada método de optimización sobre la evaluación de dos funciones de prueba elegidas para entender cómo se comportan bajo distintos escenarios de optimización.</li>
</ul>

<h2>Trabajo 2: Predictor de Riesgo Crediticio</h2>
<p>
Este proyecto es una aplicación web desarrollada en Django que permite predecir el riesgo de incumplimiento crediticio de un solicitante utilizando un modelo de red neuronal entrenado con PyTorch. El sistema toma en cuenta variables financieras y personales para calcular la probabilidad de incumplimiento y asignar un score crediticio.
</p>
<ul>
  <li><b>URL de despliegue:</b> <a href="http://3.101.66.127:8000/predictor/">http://3.101.66.127:8000/predictor/</a></li>
</ul>
<p><b>¿Cómo funciona?</b></p>
<ul>
  <li>El usuario ingresa los datos requeridos en la calculadora.</li>
  <li>El sistema preprocesa los datos y los envía al modelo de PyTorch.</li>
  <li>Se muestra el resultado: score y probabilidad de incumplimiento.</li>
  <li>Las predicciones quedan guardadas en el historial para consulta futura.</li>
</ul>
<p><b>Objetivo:</b> Ofrecer una herramienta interactiva para evaluar el riesgo crediticio de manera automática, transparente y reproducible, facilitando la toma de decisiones en procesos de crédito.</p>
<p><b>¿Cómo montarlo localmente?</b></p>
<ul>
  <li><b>Opción 1: Instalación normal</b>
    <ol>
      <li>Clona el repositorio:<br>
        <code>git clone https://github.com/druiz35/RNABI2025-1-Equipo3.git</code><br>
        <code>cd RNABI2025-1-Equipo3/Trabajo2/detector_fraude</code>
      </li>
      <li>Crea y activa un entorno virtual:<br>
        <code>python3 -m venv venv</code><br>
        <code>source venv/bin/activate</code>
      </li>
      <li>Instala las dependencias:<br>
        <code>pip install -r requirements.txt</code><br>
        <code>pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu</code>
      </li>
      <li>Aplica migraciones y ejecuta el servidor:<br>
        <code>python manage.py migrate</code><br>
        <code>python manage.py runserver</code>
      </li>
      <li>Accede a la app en <a href="http://localhost:8000/predictor/">http://localhost:8000/predictor/</a></li>
    </ol>
  </li>
  <li><b>Opción 2: Usando Docker Compose</b>
    <ol>
      <li>Ve a la carpeta del proyecto:<br>
        <code>cd RNABI2025-1-Equipo3/Trabajo2/detector_fraude</code>
      </li>
      <li>Construye y levanta los servicios:<br>
        <code>docker-compose up --build</code>
      </li>
      <li>Accede a la app en <a href="http://localhost:8000/predictor/">http://localhost:8000/predictor/</a></li>
    </ol>
  </li>
</ul>

<h2>Trabajo 3: Sistema Inteligente Integrado para Análisis de Transporte</h2>
<p>
Este proyecto implementa una solución integral de inteligencia artificial para el sector transporte, combinando tres módulos especializados en una aplicación web desarrollada con Streamlit. El sistema permite la predicción de demanda, clasificación de comportamientos de conducción y recomendación personalizada de destinos.
</p>
<ul>
  <li><b>URL de despliegue:</b> <a href="https://plankton-app-3dxgp.ondigitalocean.app/">https://plankton-app-3dxgp.ondigitalocean.app/</a></li>
</ul>

<h3>Módulos Implementados</h3>
<ul>
  <li><b>Módulo 1: Predicción de Demanda</b>
    <ul>
      <li>Utiliza modelos LSTM (Long Short-Term Memory) con TensorFlow/Keras</li>
      <li>Predice la demanda de transporte para los próximos 30 días</li>
      <li>Genera visualizaciones de tendencias y descomposición temporal</li>
      <li>Procesa datos históricos de viajes para 5 rutas principales</li>
    </ul>
  </li>
  <li><b>Módulo 2: Clasificación de Conducción</b>
    <ul>
      <li>Implementa ResNet18 con transfer learning para clasificación de imágenes</li>
      <li>Detecta 5 tipos de comportamientos: conducción segura, hablando por teléfono, enviando mensajes, girando y otras actividades</li>
      <li>Proporciona niveles de confianza y visualizaciones de probabilidades</li>
      <li>Incluye historial de predicciones y análisis estadísticos</li>
    </ul>
  </li>
  <li><b>Módulo 3: Sistema de Recomendación</b>
    <ul>
      <li>Algoritmo de filtrado colaborativo para recomendación de destinos</li>
      <li>Personalización basada en historial de usuarios y preferencias</li>
      <li>Visualización de destinos recomendados con información detallada</li>
      <li>Análisis de popularidad y características de destinos</li>
    </ul>
  </li>
</ul>

<h3>Características Técnicas</h3>
<ul>
  <li><b>Framework Principal:</b> Streamlit para interfaz web interactiva</li>
  <li><b>Deep Learning:</b> PyTorch para clasificación de imágenes</li>
  <li><b>Series Temporales:</b> TensorFlow/Keras LSTM para predicción de demanda</li>
  <li><b>Machine Learning:</b> scikit-learn para sistemas de recomendación</li>
  <li><b>Visualización:</b> Matplotlib, Seaborn y Plotly para gráficos interactivos</li>
  <li><b>Procesamiento:</b> Pandas y NumPy para análisis de datos</li>
</ul>

<h3>Funcionalidades de la Aplicación</h3>
<ul>
  <li><b>Interfaz Intuitiva:</b> Navegación por pestañas con controles sencillos</li>
  <li><b>Procesamiento en Tiempo Real:</b> Resultados inmediatos para cada módulo</li>
  <li><b>Visualizaciones Avanzadas:</b> Gráficos interactivos y métricas detalladas</li>
  <li><b>Historial de Análisis:</b> Almacenamiento y consulta de predicciones anteriores</li>
  <li><b>Análisis Estadísticos:</b> Métricas de rendimiento y tendencias temporales</li>
</ul>



<p><b>¿Cómo montarlo localmente?</b></p>
<ul>
  <li><b>Opción 1: Instalación local</b>
    <ol>
      <li>Clona el repositorio:<br>
        <code>git clone https://github.com/druiz35/RNABI2025-1-Equipo3.git</code><br>
        <code>cd RNABI2025-1-Equipo3/Trabajo3</code>
      </li>
      <li>Crea y activa un entorno virtual:<br>
        <code>python -m venv venv</code><br>
        <code>source venv/bin/activate</code> (Linux/Mac) o <code>venv\Scripts\activate</code> (Windows)
      </li>
      <li>Instala las dependencias:<br>
        <code>pip install -r requirements.txt</code>
      </li>
      <li>Ejecuta la aplicación:<br>
        <code>streamlit run app.py</code>
      </li>
      <li>Accede a la app en <a href="http://localhost:8501">http://localhost:8501</a></li>
    </ol>
  </li>
  <li><b>Opción 2: Usando Docker</b>
    <ol>
      <li>Ve a la carpeta del proyecto:<br>
        <code>cd RNABI2025-1-Equipo3/Trabajo3</code>
      </li>
      <li>Construye la imagen Docker:<br>
        <code>docker build -t streamlit-app:final .</code>
      </li>
      <li>Ejecuta el contenedor:<br>
        <code>docker run -p 8501:8501 streamlit-app:final</code>
      </li>
      <li>Accede a la app en <a href="http://localhost:8501">http://localhost:8501</a></li>
    </ol>
  </li>
</ul>

<p><b>Objetivo:</b> Proporcionar una herramienta integral para la toma de decisiones en el sector transporte, combinando análisis predictivo, clasificación automática y personalización de servicios para optimizar la operación y mejorar la experiencia del usuario.</p>

<h1>Miembros del Equipo</h1>
<h3>Leonardo Federico Corona Torres</h3>
Estudiante de Estadística con una fuerte pasión por la consultoria, el análisis de datos, el desarrollo de software y la inteligencia artificial.
<h3>David Escobar Ruiz</h3>
Estudiante de ingeniería de sistemas con aspiraciones de desarrollar una carrera profesional en el área de la ingeniería de machine learning y ganar experticia en Agentic AI y NLP.
<h3>Johan Sebastian Robles Rincón</h3>
Estudiante de sistemas e informática con enfoque en aplicar soluciones robustas por el desarrollo Backend e infraestructura.
<h3>Sebastián Soto Arcila</h3>
Estudiante de ingeniería en sistemas con enfoque en el desarrollo full stack y apacionado por el tema de toma de decisiones basadas en datos.
