import streamlit as st
from notebooks.StoryAgent import StoryAgent
from langchain_core.messages import AIMessage, HumanMessage

# Función para cargar el modelo
@st.cache_resource
def load_model():
    return StoryAgent()

storyAgent = load_model()

# Título de la app
st.title("📖 Generador de Prompt para Historia")

# Campos de entrada

# Opciones predefinidas para género y tono
generos = [
     "Fantasía", "Misterio", "Romance",
     "Terror", "Ciencia Ficción", "Comedia",
     "Aventura"
]

tonos = [
    "Oscuro", "Esperanzador", "Melancólico", "Trágico", 
    "Cómico", "Serio", "Sarcástico", "Inspirador", "Reflexivo"
]

# Listas desplegables
genero = st.selectbox("🎭 Género literario", generos)
tono = st.selectbox("🥵 Tono", tonos)

# Longitud con radio buttons
longitud_opciones = {
    "Corta": "Cuento corto (menos de 1000 palabras)",
    "Media": "Relato medio (1000 a 3000 palabras)",
    "Larga": "Novela corta o larga (más de 3000 palabras)"
}
longitud_seleccionada = st.radio("📏 Longitud", list(longitud_opciones.keys()))
longitud = longitud_opciones[longitud_seleccionada]

# Entradas adicionales
periodo_de_tiempo = st.text_input("⌚ Período de tiempo", placeholder="Ej. Edad Media, Futuro lejano...")
ubicacion = st.text_input("🗺 Ubicación", placeholder="Ej. Nueva York, un planeta lejano...")
atmosfera = st.text_area("🌍 Atmósfera", placeholder="Ej. tensa, misteriosa, nostálgica...")
conflictos = st.text_area("💪 Conflictos", placeholder="Describe el conflicto principal o varios...")
obstaculos = st.text_area("🚧 Obstáculos", placeholder="Obstáculos que los personajes deben enfrentar...")
resolucion = st.text_area("🏁 Resolución", placeholder="¿Cómo se resuelve el conflicto?")
personajes = st.text_area("👥 Personajes", placeholder="Nombres y descripciones breves de los personajes...")

# Botón para generar el prompt
if st.button("Generar la historia"):
    campos_vacios = []

    if not periodo_de_tiempo:
        campos_vacios.append("Período de tiempo")
        
    if not ubicacion:
        campos_vacios.append("Ubicación")
        
    if not atmosfera:
        campos_vacios.append("Atmósfera")
        
    if not conflictos:
        campos_vacios.append("Conflictos")
        
    if not obstaculos:
        campos_vacios.append("Obstáculos")
        
    if not resolucion:
        campos_vacios.append("Resolución")
        
    if not personajes:
        campos_vacios.append("Personajes")

    if campos_vacios:
        user_prompt = f"Recomienda posibles respuestas para los siguientes campos necesarios para generar una historia: {campos_vacios}"
        input_message = {
            "role": "user",
            "input": user_prompt,
            "chat_history": storyAgent.chat_history
        }
        response = storyAgent.agent_executor.invoke(input_message)
        storyAgent.chat_history.append(HumanMessage(content=user_prompt))
        storyAgent.chat_history.append(AIMessage(content=response["output"]))
        st.subheader("❌ Campos Faltantes:")
        st.write(response["output"])

    else:
        user_prompt = storyAgent.build_prompt(genero, tono, longitud, personajes, periodo_de_tiempo, ubicacion, atmosfera, conflictos, obstaculos, resolucion)
        input_message = {
            "role": "user",
            "input": user_prompt,
            "chat_history": storyAgent.chat_history
        }
        response = storyAgent.agent_executor.invoke(input_message)
        storyAgent.chat_history.append(HumanMessage(content=user_prompt))
        storyAgent.chat_history.append(AIMessage(content=response["output"]))
        st.subheader("📝 Historia generada:")
        st.write(response["output"])