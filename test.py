from langchain.messages import HumanMessage
from langchain_core.utils.uuid import uuid7
from main import agent

# Configuration for this conversation thread
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}


# Turn 1: Initial message - starts with warranty_collector step
print("=== Turn 1: Warranty Collection ===")
result = agent.invoke(
    {"messages": [HumanMessage("Hi, my phone screen is cracked")]}, config=config
)

for msg in result["messages"]:
    msg.pretty_print()


# Turn 2: User responds about warranty
print("\n=== Turn 2: Warranty Response ===")
result = agent.invoke(
    {"messages": [HumanMessage("Yes, it's still under warranty")]}, config=config
)

for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result['current_step']}")


# Turn 3: User describes the issue
print("\n=== Turn 3: Issue description ===")
result = agent.invoke(
    {"messages": [HumanMessage("The screen is physically cracked from dropping it.")]},
    config=config,
)

for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result['current_step']}")


# Turn 4: Resolution
print("\n=== Turn 4: Resolution ===")
result = agent.invoke({"messages": [HumanMessage("What should I do?")]}, config=config)

for msg in result["messages"]:
    msg.pretty_print()

print(f"Current step: {result['current_step']}")


# Turn 5: Go back for correction
print("\n=== Turn 5: Go back for correction ===")
result = agent.invoke(
    {
        "messages": [
            HumanMessage("Actually, I made a mistake - my device is out of warranty")
        ]
    },
    config=config
)
# Agent will call go_back_to_warranty and restart warranty verification step.

for msg in result["messages"]:
    msg.pretty_print()
