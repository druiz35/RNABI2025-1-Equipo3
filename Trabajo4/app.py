import streamlit as st
from notebooks.StoryAgent import StoryAgent
from langchain_core.messages import AIMessage, HumanMessage

# Función para cargar el modelo
@st.cache_resource
def load_model():
    return StoryAgent()

storyAgent = load_model()

# Título de la app
st.title("Generador de Prompt para Historia")

# Campos de entrada
genero = st.text_input("Género literario", placeholder="Ej. Ciencia ficción, Romance, Fantasía...")
tono = st.text_input("Tono", placeholder="Ej. Trágico, esperanzador, oscuro...")
longitud = st.text_input("Longitud", placeholder="Ej. Cuento corto, Novela, 1000 palabras...")
periodo_de_tiempo = st.text_input("Período de tiempo", placeholder="Ej. Edad Media, Futuro lejano...")
ubicacion = st.text_input("Ubicación", placeholder="Ej. Nueva York, un planeta lejano...")
atmosfera = st.text_input("Atmósfera", placeholder="Ej. Tensa, misteriosa, nostálgica...")
conflictos = st.text_area("Conflictos", placeholder="Describe el conflicto principal o varios...")
obstaculos = st.text_area("Obstáculos", placeholder="Obstáculos que los personajes deben enfrentar...")
resolucion = st.text_area("Resolución", placeholder="¿Cómo se resuelve el conflicto?")
personajes = st.text_area("Personajes", placeholder="Nombres y descripciones breves de los personajes...")

# Botón para generar el prompt
if st.button("Generar Prompt"):
    user_prompt = storyAgent.build_prompt(genero, tono, longitud, personajes, periodo_de_tiempo, ubicacion, atmosfera, conflictos, obstaculos, resolucion)
    input_message = {
        "role": "user",
        "input": user_prompt,
        "chat_history": storyAgent.chat_history
    }
    response = storyAgent.agent_executor.invoke(input_message)
    storyAgent.chat_history.append(HumanMessage(content=user_prompt))
    storyAgent.chat_history.append(AIMessage(content=response["output"]))
    st.subheader("Historia generada:")
    st.write(response["output"])