from dotenv import load_dotenv
from langchain_core.tools import tool
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_react_agent
from langchain.tools import BaseTool

load_dotenv()

api_key = os.environ.get('GROQ_API_KEY')

def get_gmail_service():
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_desktop.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        
    service = build('gmail', 'v1', credentials=creds)
    return service

def get_mail(duration: str = "3d"):
    """
    Fetch emails from Gmail received in the last n days.
    Args:
        duration (str): Duration string like '3d' for 3 days.
    """
    service = get_gmail_service()

    user_id = 'me'
    query = f'newer_than:{duration}'
    mail_data = []
    page_token = None

    try:
        while True:
            results = service.users().messages().list(
                userId=user_id, 
                q=query, 
                pageToken=page_token
            ).execute()
            
            messages = results.get('messages', [])
            if not messages:
                break

            for message in messages:
                try:
                    msg = service.users().messages().get(
                        userId=user_id, 
                        id=message['id'], 
                        format='full',  
                        metadataHeaders=['From', 'Subject', 'Date']
                    ).execute()
                    mail_data.append(msg)
                except HttpError as e:
                    print(f"Error fetching message {message['id']}: {e}")

            page_token = results.get('nextPageToken')
            if not page_token:
                break

        if not mail_data:
            print('No messages found in the last 3 days.')

        return mail_data

    except HttpError as e:
        print(f"Gmail API error: {e}")
        return []


model = init_chat_model(model_provider="groq", model="openai/gpt-oss-20b", api_key=api_key).bind_tools([get_mail])


query = "Summarize the emails I received in the last 2 days."
