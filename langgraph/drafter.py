from typing import Annotated, Sequence, TypedDict, cast, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()

document_history: List[str] = []

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str)-> str:
    """Create a new document version using the provided content.

    This preserves previous versions in memory while making the provided
    content the latest version. The tool returns the latest version content.
    """
    global document_history
    document_history.append(content)
    return content

@tool
def save(filename: str)-> str:
    """Save the document to a file"""
    global document_history
    if not filename.endswith(".txt"):
        filename += ".txt"
    
    try:
        latest = document_history[-1] if document_history else ""
        with open(filename, "w") as f:
            f.write(latest)
        print(f"Document saved to {filename}")
        return f"Document saved to {filename}"
    except Exception as e:
        return f"Error saving document: {e}"

tools = [update, save]

model = ChatOpenAI(api_key="None", base_url="http://localhost:1234/v1").bind_tools(tools)
    
def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
    You are Drafter, an AI assistant that helps user draft documents.
    Your goal is to help the user draft a document by using the available tools to update and save the document as needed.

    - If the user wants to update or modify content in the document, use the 'update' tool.
    - If the user wants to save the document, use the 'save' tool.
    - Make sure to show current document state after modification

    Current Document Content: {document_history[-1] if document_history else ""}
    """
    )

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you write to create\n"
        user_message = HumanMessage(content =  user_input)
    else:
        user_input = input("\n What would you like to do with the document?\n>")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    prior = list(state["messages"]) if state.get("messages") is not None else []
    all_messages = [system_prompt] + prior + [user_message]

    response = model.invoke(all_messages)

    print(f"AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""

    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content = message.content
            if isinstance(content, str):
                lc = content.lower()
                if ("saved" in lc) and ("document" in lc):
                    return "end"
    
    return "continue"

def print_message(messages):
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"TOOL RESULT: {message.content}")
    

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {
    "continue": "agent",
    "end": END,
})


app =graph.compile()

def run_drafter_agent():
    print("\n ---------------------- Drafter ------------------")

    state = cast(AgentState, {"messages": []})

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])
    
    print("\n --------------------- Drafter Done ------------------------")

if __name__  == "__main__":
    run_drafter_agent()