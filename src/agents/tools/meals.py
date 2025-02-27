from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

@tool
def set_meal_tool(meal: str, date: str, special_config_param: RunnableConfig):
    """
    Sets the meal plan for a specific date.

    Args:
        meal (str): The name of the meal.
        date (str): The date of the meal plan.
    """
    return f"Meal plan created and sent to the provider: {meal} for {date} by team member {special_config_param['configurable']['user']}"