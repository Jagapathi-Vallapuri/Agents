from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-20b", temperature=0.3)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAgent response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
