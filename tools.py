from custom_state import SupportState
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Literal


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition
    to the issue classification."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["software", "hardware"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to the
    resolution specialist."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist."""
    # In a real system, this would create a ticket, notify staff etc.
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided: {solution}"


# Define prompts as constants for easy reference
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping
with device issues.

CURRENT STAGE: Warranty verification

At this step you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response and
move to the next step

Be conversational and friendly. Don't ask multiple questions at once.
"""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping
with the device issues.


CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}.

At this step you need to:
1. Ask the customer to describe their issue
2. Determine if it's a hardware issue (physical damage, broken parts)
or software issue (app crashes, performance)
3. Use record_issue_type to record the classification and move to the next step

If unclear, ask clarifying questions before classifying
"""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping
with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step you need to:
1. For software issues: provide troubleshooting steps using provide_solution
2. For hardware issues:
    - IF IN WARRANTY: explain warranty repair process using provide_solution
    - IF OUT OF WARRANTY: escalate_to_human for paid repair options

Be specific and helpful in your solutions.
"""


# Step configuration: maps step names to (prompt, tools, required_state)
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": []
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"]
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"]
    }
}
