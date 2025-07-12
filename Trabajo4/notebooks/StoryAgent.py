import getpass
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent

# from google import genai

from pprint import pprint


class StoryAgent:
    MAIN_PROMPT = "Eres un generador de narrativas que generará una historia a partir de la siguiente información dada por el usuario:\n"
    "* Género: La historia que puedes generar puede pertenecer a alguno de los siguientes géneros: Fantasía, Misterio, Romance, Terror, Ciencia Ficción, Comedia, Aventura."
    "* Tono: El tono de la historia en general. Puede ser uno de los siguientes: Humorístico, Oscuro, Caprichoso, Dramático, Satírico."
    "* Longitud de la historia: Puede ser corta (400 palabras), mediana (600 palabras) o larga (800 palabras)."
    "* Personajes: El usuario puede agregar múltiples personajes a su historia, y por cada uno deberá especificar lo siguiente: Nombre, Rol, Rasgos de Personalidad, Relaciones."
    "* Escenario: El usuario deberá especificar lo siguiente con respecto al escenario: Período de Tiempo, Ubicación, Atmósfera."
    "* Elementos de trama: El usuario deberá especificar lo siguiente con respecto a cada uno de los elementos de la trama de la historia que desea: Tipo de Conflicto, Obstáculos, Estilo de Resolución."

    STRUCTURE_PROMPT = "La estructura narrativa de la historia a generar debe incluir inicio, nudo y desenlace.\n"
    "La historia generada solo debe contener el título y la historia. No agregues nada más.\n"

    GENRE_PROMPT = "Ten en cuenta los siguientes tips para cada uno de los géneros de las historias:" \
                   "* Fantasía: Haz énfasis en la construcción del mundo antes de introducir a los personajes." \
                   "* Misterio: Usa elementos narrativos de este género como los presagios y las pistas." \
                   "* Romance: Puedes involucrar un pequeño conflicto seguido de una reconciliación antes de terminar las historias de este género." \
                   "* Terror: Utiliza finales ambiguos e inciertos que dejen al lector en suspenso." \
                   "* Ciencia Ficción: Intenta utilizar conceptos científicos reales con una ligera alteración." \
                   "* Comedia: Utiliza el entorn a tu favor para hacer reír al lector." \
                   "* Aventura: Incluye mundos y sitios sin descubrir junto con artefactos perdidos y arcaicos.\n"
    
    FULL_PROMPT = MAIN_PROMPT + STRUCTURE_PROMPT + GENRE_PROMPT

    def __init__(self):
        self.load_env()
        self.model = init_chat_model(
            "gemini-2.5-flash",
            model_provider="google_genai"
        )
        self.tools = list()
        self.set_main_prompt()
        self.set_agent()
        # self.create_tools()
        self.chat_history = list()

    def build_prompt(self, genero, tono, longitud, personajes, periodo_de_tiempo, ubicacion, atmosfera, conflictos, obstaculos, resolucion):
        prompt = f"Necesito que crees un título y una historia a partir de la siguiente información: \n"
        prompt += f"* Género literario {genero}\n"
        prompt += f"* Tono: {tono}\n"
        prompt += f"* Longitud: {longitud}\n"
        prompt += f"* Período de tiempo: {periodo_de_tiempo}\n"
        prompt += f"* Ubicación: {ubicacion}\n"
        prompt += f"* Atmósfera: {atmosfera}\n"
        prompt += f"* Conflictos: {conflictos}\n"
        prompt += f"* Obstáculos: {obstaculos}\n"
        prompt += f"* Resolución: {resolucion}\n"
        prompt += f"* Personajes: {personajes}"
        return prompt

    def load_env(self):
        load_dotenv()
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

    def set_main_prompt(self):
        self.main_prompt = ChatPromptTemplate([
            ("system", StoryAgent.FULL_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

    def set_agent(self):
        self.agent = create_tool_calling_agent(
            self.model,
            self.tools,
            self.main_prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False
        )

    def test_run(self):
        genero = input("Género literario: ")
        tono = input("Tono: ")
        longitud = input("Longitud: ")
        personajes = input("Personajes: ")
        periodo_de_tiempo = input("Período de tiempo: ")
        ubicacion = input("Ubicación: ")
        atmosfera = input("Atmósfera: ")
        conflictos = input("Conflictos: ")
        obstaculos = input("Obstáculos: ")
        resolucion = input("Resolución: ")
        user_prompt = self.build_prompt(genero, tono, longitud, personajes, periodo_de_tiempo, ubicacion, atmosfera, conflictos, obstaculos, resolucion)
        input_message = {
            "role": "user",
            "input": user_prompt,
            "chat_history": self.chat_history
        }
        response = self.agent_executor.invoke(input_message)
        self.chat_history.append(HumanMessage(content=user_prompt))
        self.chat_history.append(AIMessage(content=response["output"]))
        #print(response["output"])
        #print(len(response["output"].split()))
        return response["output"]

if __name__ == "__main__":
    story_agent = StoryAgent()
    print(story_agent.test_run())
