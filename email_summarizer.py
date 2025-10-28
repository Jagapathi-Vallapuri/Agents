from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from typing import Union, TypedDict, List, Optional, cast
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]

def get_gmail_service():
    """Create an authenticated Gmail API service.

    - Uses token.json for cached credentials
    - If refresh fails (e.g., invalid_grant), deletes token.json and re-authenticates
    - Uses desktop client credentials by default (override with GOOGLE_OAUTH_CLIENT_SECRET)
    """
    creds = None

    token_path = os.getenv("GOOGLE_GMAIL_TOKEN", "token.json")
    client_secret_path = os.getenv(
        "GOOGLE_OAUTH_CLIENT_SECRET", "client_secret_desktop.json"
    )

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes=SCOPES)

    def _save(creds_obj):
        with open(token_path, "w") as token:
            token.write(creds_obj.to_json())

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                try:
                    if os.path.exists(token_path):
                        os.remove(token_path)
                finally:
                    creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        _save(creds)

    service = build("gmail", "v1", credentials=creds)
    return service

@tool("get_mails")
def get_mails(query: str) -> List[dict]:
    """
        input: query string in Gmail search format (avoid backticks and invalid tokens)

        examples of valid filters:
            - from:someone@example.com
            - subject:"Meeting notes"
            - has:attachment
            - after:2025/10/27 before:2025/10/28
            - newer_than:2d   (days/months/years supported: d, m, y; hours are NOT supported)

        Notes:
            - Gmail doesn't support hour-level windows in the query string (e.g., newer_than:1h is invalid).
                If you need last N hours, pass a generic query (or empty string) and the tool will filter
                client-side by internalDate.

        Fetch INBOX emails but only retrieve minimal metadata from Gmail (not full bodies).

        Returns a list of dicts per message with keys: id, snippet, subject, from, date.
    """
    service = get_gmail_service()
    user_id = "me"
    results_list: List[dict] = []

    try:
        next_page_token = None
        while True:
            results = (
                service.users()
                .messages()
                .list(
                    userId=user_id,
                    labelIds=["INBOX"],
                    q=query,
                    pageToken=next_page_token,
                    maxResults=5,
                    fields="messages(id),nextPageToken",
                )
                .execute()
            )

            messages = results.get("messages", [])
            if not messages:
                break

            for m in messages:
                msg = (
                    service.users()
                    .messages()
                    .get(
                        userId=user_id,
                        id=m["id"],
                        format="metadata",
                        metadataHeaders=["Subject", "From", "Date"],
                    )
                    .execute()
                )
                headers = {
                    h.get("name", ""): h.get("value", "")
                    for h in msg.get("payload", {}).get("headers", [])
                }
                results_list.append(
                    {
                        "id": m["id"],
                        "snippet": msg.get("snippet", ""),
                        "subject": headers.get("Subject", ""),
                        "from": headers.get("From", ""),
                        "date": headers.get("Date", ""),
                    }
                )

            next_page_token = results.get("nextPageToken")
            if not next_page_token:
                break

    except HttpError as e:
        print(f"An error occurred while fetching emails: {e}")
        return []

    return results_list

tools = [get_mails]


def agent(state: AgentState) -> AgentState:
    """Agent node that can optionally call tools.

    Binds tools to the LLM so it can decide to emit tool_calls when needed.
    If no tool is needed, it will just return a normal chat response.
    """
    

    llm = ChatGroq(model = "llama-3.3-70b-versatile",temperature=0.2, api_key=SecretStr(os.getenv("GROQ_API_KEY", "not-needed"))).bind_tools(tools, tool_choice="auto")

    system_prompt = SystemMessage(
        content=(
            "You are a helpful assistant. Use tools ONLY when necessary to answer "
            "the user's question; otherwise, reply directly as a normal chatbot."
        )
    )

    ai = cast(AIMessage, llm.invoke([system_prompt] + state["messages"]))
    return {"messages": state["messages"] + [ai]}


def should_continue(state: AgentState) -> str:
    if not state["messages"]:
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def build_app():
    """Compile the LangGraph app with conditional tool usage."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_chat():
    """Simple interactive loop: uses tools only if the LLM asks for them."""
    app = build_app()
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]] = []

    print("Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        messages.append(HumanMessage(content=user))
        result = app.invoke({"messages": messages})
        messages = result["messages"]
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai:
            print(f"Assistant: {last_ai.content}")


if __name__ == "__main__":
    run_chat()
