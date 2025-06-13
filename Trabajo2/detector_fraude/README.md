## Trabajo 2: Predictor de Riesgo Crediticio

Este proyecto es una aplicación web desarrollada en Django que permite predecir el riesgo de incumplimiento crediticio de un solicitante utilizando un modelo de red neuronal entrenado con PyTorch. El sistema toma en cuenta variables financieras y personales para calcular la probabilidad de incumplimiento y asignar un score crediticio.

- **URL de despliegue:** http://3.101.66.127:8000/predictor/

### ¿Cómo funciona?
- El usuario ingresa los datos requeridos en la calculadora.
- El sistema preprocesa los datos y los envía al modelo de PyTorch.
- Se muestra el resultado: score y probabilidad de incumplimiento.
- Las predicciones quedan guardadas en el historial para consulta futura.

### Objetivo
El objetivo es ofrecer una herramienta interactiva para evaluar el riesgo crediticio de manera automática, transparente y reproducible, facilitando la toma de decisiones en procesos de crédito.

### ¿Cómo montarlo localmente?

#### Opción 1: Instalación normal
1. Clona el repositorio:
   ```bash
   git clone https://github.com/druiz35/RNABI2025-1-Equipo3.git
   cd RNABI2025-1-Equipo3/Trabajo2/detector_fraude
   ```
2. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
4. Aplica migraciones y ejecuta el servidor:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```
5. Accede a la app en [http://localhost:8000/predictor/](http://localhost:8000/predictor/)

#### Opción 2: Usando Docker Compose
1. Ve a la carpeta del proyecto:
   ```bash
   cd RNABI2025-1-Equipo3/Trabajo2/detector_fraude
   ```
2. Construye y levanta los servicios:
   ```bash
   docker-compose up --build
   ```
3. Accede a la app en [http://localhost:8000/predictor/](http://localhost:8000/predictor/) 