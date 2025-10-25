import os
from dotenv import load_dotenv
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(api_key="", base_url="http://127.0.0.1:1234/v1", temperature=0.3)

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAgent response: {response.content}")
    print(f"Current conversation history: {[msg for msg in state['messages']]}")
    return state



graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

memory = []

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    memory.append(HumanMessage(content=user_input))
    res = agent.invoke({"messages": memory})
    memory = res["messages"]
    user_input = input("Enter: ")
