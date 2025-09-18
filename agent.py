from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import os

def create_agent():
    search = DuckDuckGoSearchAPIWrapper()
    search_tool = Tool(
        name="Search",
        func=search.run,
        description="Use to search for current events or information from the web"
    )

    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",  # you can also use "gemini-pro-vision" for images
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent="chat-zero-shot-react-description",
        verbose=True
    )
    return agent
