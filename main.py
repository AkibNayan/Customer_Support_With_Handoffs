from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from components import model
from custom_state import SupportState
from middleware import apply_step_config
from tools import (
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human
)

all_tools = [record_warranty_status, record_issue_type,
             provide_solution, escalate_to_human]


# Create the agent with step-based configuration
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver()
)
