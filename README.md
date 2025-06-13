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
  <li>Implementar en el lenguaje R los siguientes métodos de optimización: Descenso de gradiente, Algoritmos evolutivos, 
      Colonias de hormigas, Enjambre de partículas, Evolución diferencial.
  </li>
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
<h3>Trabajo 3: Aplicaciones en sistemas de recomendación e imágenes</h3>
<aside>⚠️Pendiente</aside>
<h3>Trabajo 4: Aplicaciones de grandes modelos de lenguaje</h3>
<aside>⚠️Pendiente</aside>

<h1>Miembros del Equipo</h1>
<h3>Leonardo Federico Corona Torres</h3>
Estudiante de Estadística con una fuerte pasión por la consultoria, el análisis de datos, el desarrollo de software y la inteligencia artificial.
<h3>David Escobar Ruiz</h3>
Estudiante de ingeniería de sistemas con aspiraciones de desarrollar una carrera profesional en el área de la ingeniería de machine learning y ganar experticia en Agentic AI y NLP.
<h3>Johan Sebastian Robles Rincón</h3>
Estudiante de sistemas e informática con enfoque en aplicar soluciones robustas por el desarrollo Backend e infraestructura.
<h3>Sebastián Soto Arcila</h3>
Estudiante de ingeniería en sistemas con enfoque en el desarrollo full stack y apacionado por el tema de toma de decisiones basadas en datos.
