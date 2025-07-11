import getpass
import os
from typing import Annotated, List
from dotenv import dotenv_values, load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
#from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent

from pprint import pprint


class StoryAgent:
    MAIN_PROMPT = "Eres un asistente de creación de narrativas que recibirá la siguiente información de parte del usuario para poder generar una historia:"
    "* Género: La historia que puedes generar puede pertenecer a alguno de los siguientes géneros: Fantasía, Misterio, Romance, Terror, Ciencia Ficción, Comedia, Aventura."
    "* Tono: El tono de la historia en general. Puede ser uno de los siguientes: Humorístico, Oscuro, Caprichoso, Dramático, Satírico."
    "* Longitud de la historia: Puede ser corta (400 palabras), mediana (600 palabras) o larga (800 palabras)."
    "* Personajes: El usuario puede agregar múltiples personajes a su historia, y por cada uno deberá especificar lo siguiente: Nombre, Rol, Rasgos de Personalidad, Relaciones."
    "* Escenario: El usuario deberá especificar lo siguiente con respecto al escenario: Período de Tiempo, Ubicación, Atmósfera."
    "* Elementos de trama: El usuario deberá especificar lo siguiente con respecto a cada uno de los elementos de la trama de la historia que desea: Tipo de Conflicto, Obstáculos, Estilo de Resolución."
    "La estructura narrativa de la historia a generar debe incluir inicio, nudo y desenlace."
    
    def __init__(self):
       self.load_env()
       self.model = init_chat_model(
          "gemini-2.5-flash",
          model_provider="google_genai"
       )
       self.tools = list()
       #self.create_tools()
       self.set_prompts()
       self.set_agent()
       self.chat_history = list()
       


    def load_env(self):
        load_dotenv()
        if not os.environ.get("GOOGLE_API_KEY"):
          os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    
    def set_prompts(self):
       self.main_prompt = ChatPromptTemplate([(
            "system", 
            "Eres un asistente de creación de narrativas que recibirá la siguiente información de parte del usuario para poder generar una historia:"
            "* Género: La historia que puedes generar puede pertenecer a alguno de los siguientes géneros: Fantasía, Misterio, Romance, Terror, Ciencia Ficción, Comedia, Aventura."
            "* Tono: El tono de la historia en general. Puede ser uno de los siguientes: Humorístico, Oscuro, Caprichoso, Dramático, Satírico."
            "* Longitud de la historia: Puede ser corta (400 palabras), mediana (600 palabras) o larga (800 palabras)."
            "* Personajes: El usuario puede agregar múltiples personajes a su historia, y por cada uno deberá especificar lo siguiente: Nombre, Rol, Rasgos de Personalidad, Relaciones."
            "* Escenario: El usuario deberá especificar lo siguiente con respecto al escenario: Período de Tiempo, Ubicación, Atmósfera."
            "* Elementos de trama: El usuario deberá especificar lo siguiente con respecto a cada uno de los elementos de la trama de la historia que desea: Tipo de Conflicto, Obstáculos, Estilo de Resolución."
            "La estructura narrativa de la historia a generar debe incluir inicio, nudo y desenlace."
            ),
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
          verbose=False,
          #memory=memory
       )
    
    def run(self):
       while True:
            user_input = input(">")
            input_message = {
                "role": "user",
                "input": user_input,
                "chat_history": self.chat_history
            }
            response = self.agent_executor.invoke(
                input_message
            )
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response["output"]))
            pprint(response["output"])

if __name__ == "__main__":
   story_agent = StoryAgent()
   story_agent.run()