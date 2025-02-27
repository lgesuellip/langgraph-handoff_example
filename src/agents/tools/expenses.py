from typing import List
from pydantic import BaseModel
from langchain_core.tools import tool
import logging
from langchain_core.runnables import RunnableConfig
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

class Expense(BaseModel):
    expense_type: str
    date: str
    total_value: float
    state: str

class UserExpenses(BaseModel):
    expenses: List[Expense]

@tool
def get_expenses_tool(special_config_param: RunnableConfig) -> str:
    """
    Retrieves all pending expenses
    """

    # Mock implementation for testing purposes
    mock_expenses = {
        "user1": UserExpenses(
            expenses=[
                Expense(
                    expense_type="food",
                    date="2023-05-01",
                    total_value=50.0,
                    state="pending"
                ),
                Expense(
                    expense_type="coffee",
                    date="2023-05-02",
                    total_value=5.0,
                    state="pending"
                )
            ]
        )
    }
    return f"Expenses retrieved: {mock_expenses[special_config_param['configurable']['user']].expenses}"