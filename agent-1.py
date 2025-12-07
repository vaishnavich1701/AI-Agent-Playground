from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import io
import sys
import re
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a News Agent that uses DuckDuckGo to search for news stories
news_agent = Agent(
    name="News Agent",
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=False,
    markdown=False,
)

# Create a Finance Agent that uses YFinance to fetch financial data
finance_agent = Agent(
    name="Finance Agent",
    description="You are an expert finance analyst providing detailed stock insights.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    show_tool_calls=False,
    markdown=False,
)

def clean_response(response: str) -> str:
    """
    Function to clean the response from the agent by removing unwanted parts like borders, extra spaces, 
    and ANSI escape sequences.
    """
    # Remove borders and unnecessary lines (like ┏━ and ┛)
    pattern = r"(┏━.*?━┛|┛|┏━.*?━┛|\n)+"
    
    # Remove any remaining ANSI escape sequences (color codes)
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    cleaned_response = re.sub(ansi_escape, '', response)  # Remove color codes

    # Remove borders (┏━, ┛, ┏━, etc.)
    cleaned_response = re.sub(pattern, "", cleaned_response).strip()

    return cleaned_response


@app.get("/news")
async def get_news(query: str):
    """
    Endpoint to get a news report based on the query.
    Example: GET /news?query=latest+New+York+news
    """
    try:
        # Capture the printed output from the news agent
        old_stdout = sys.stdout                     # Save current stdout
        sys.stdout = mystdout = io.StringIO()         # Redirect stdout to StringIO
        news_agent.print_response(query, stream=False)  # This prints to our StringIO
        response = mystdout.getvalue()               # Capture the printed output
        sys.stdout = old_stdout                      # Restore stdout

        # Clean the response using the clean_response function
        cleaned_response = clean_response(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"agent": "News Agent", "query": query, "response": cleaned_response}

@app.get("/finance")
async def get_finance(query: str):
    """
    Endpoint to get financial data based on the query.
    Example: GET /finance?query=Apple+stock+performance
    """
    try:
        # Capture the printed output from the finance agent
        old_stdout = sys.stdout                     # Save current stdout
        sys.stdout = mystdout = io.StringIO()         # Redirect stdout to StringIO
        finance_agent.print_response(query, stream=False)  # This prints to our StringIO
        response = mystdout.getvalue()               # Capture the printed output
        sys.stdout = old_stdout                      # Restore stdout

        # Clean the response using the clean_response function
        cleaned_response = clean_response(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"agent": "Finance Agent", "query": query, "response": cleaned_response}


if __name__ == "__main__":
    import uvicorn
    # Replace 'main:app' with 'filename:app' if you name your file differently.
    uvicorn.run("agent-1:app", host="localhost", port=8000, reload=True)
