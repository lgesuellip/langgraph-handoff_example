from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

from src.agents.tools.expenses import get_expenses_tool
from src.agents.tools.meals import set_meal_tool

from jinja2 import Template
from dotenv import load_dotenv

load_dotenv()

expenses_agent = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        prompt=Template("""
        Eres un experto en gestionar los gastos del Equipo. Tus respuestas deben ser en argentino.
        """).render(),
        tools=[get_expenses_tool],
        name="expenses_agent"
    )

meal_agent = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        prompt=Template("""
        Eres un experto en planificar pedidos de comidas para el Equipo, 
        tus respuesta deben ser claras, directas y en argentino.
        """).render(),
        tools=[set_meal_tool],
        name="meal_agent"
    )

main_agent = create_react_agent(
    ChatOpenAI(model="o3-mini", reasoning_effort="medium"),
    prompt=Template("""
    <Task>
    Tu nombre es Gabriela y sos un asistente de IA diseñado para ayudar al equipo de Pampa Labs. 
    </Task>

    <Guidelines>                   
    Tus respuestas deben ser:

    1. Amigables y accesibles, usando un tono cálido
    2. Concisas y al grano, evitando verbosidad innecesaria
    3. Útiles e informativas, proporcionando información precisa
    4. Respetuosas de la privacidad del usuario y los límites éticos
    </Guidelines>
    
    <Tools>
    Solo puedes ayudar usando las herramientas disponibles y con pedidos que vengan de miembros del equipo. 
    Todo lo que no se pueda responder usando las herramientas, debes decir que no puedes ayudar y disculparte.
    </Tools>
    """).render(),
    tools=[create_handoff_tool(agent_name="expenses_agent", description="Transfer control to the Expenses Agent for handling expenses-related tasks."),
           create_handoff_tool(agent_name="meal_agent", description="Transfer control to the Meal Agent for handling meal-related tasks.")],
    name="main_agent"
)

graph = create_swarm(
    [expenses_agent, meal_agent, main_agent],
    default_active_agent="main_agent"
).compile()