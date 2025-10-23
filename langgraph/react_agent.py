import os 
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(x: int, y: int):
    """This is an addition function to add 2 numbers"""
    return x + y

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are an ai assistant, please answer my query to the best of your ability.")

    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}



def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
tools = [add, subtract, multiply]

model = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)


graph = StateGraph(AgentState)

graph.add_node("agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})

graph.add_edge("tools", "agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        msg = s["messages"][-1]
        
        if isinstance(msg, tuple):
            print(msg)
        else:
            msg.pretty_print()

inputs = {"messages": [("user","Add 40 + 12 and then multiply the result by 6. Also tell me a dad joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))