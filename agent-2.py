from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware 
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import io
import sys
import re
import json
from typing import Iterator
from agno.tools.dalle import DalleTools
from agno.utils.common import dataclass_to_dict

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

# Create an Image Agent that uses DALL-E to generate images
image_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[DalleTools()],
    description="You are an AI agent that can create images using DALL-E.",
    instructions=[
        "When the user asks you to create an image, use the DALL-E tool to create an image.",
        "The DALL-E tool will return an image URL.",
        "Return the image URL in your response in the following format: `![image description](image URL)`",
    ],
    markdown=True,
)


def clean_response(response: str) -> str:
    """
    Cleans the agent response by:
    - Removing ANSI escape sequences.
    - Splitting the response into lines.
    - Removing leading/trailing border characters from each line.
    - Skipping lines that are purely decorative (e.g. headings like "Message" or "Response").
    - Joining the remaining lines with newlines.
    """
    # Remove ANSI escape sequences
    response = re.sub(r'\x1b\[[0-9;]*m', '', response)
    # Split the response into lines
    lines = response.splitlines()
    cleaned_lines = []
    for line in lines:
        # Remove leading and trailing border characters and whitespace
        cleaned_line = re.sub(r'^[┏┗┃━\s]+', '', line)
        cleaned_line = re.sub(r'[┏┗┃━\s]+$', '', cleaned_line)
        # Skip lines that start with "Message" or "Response" (these are decorative)
        if re.match(r'^(Message|Response)', cleaned_line):
            continue
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    # Join the cleaned lines preserving paragraph breaks
    return "\n".join(cleaned_lines).strip()

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

@app.get("/generate_image")
async def generate_image(prompt: str):
    """
    Endpoint to generate an image using the image agent.
    Accepts a query parameter "prompt" for the image description.
    Returns a shortened response containing only the final result with the image URL.
    """
    try:
        final_result = None
        run_stream: Iterator = image_agent.run(prompt, stream=True, stream_intermediate_steps=True)
        for chunk in run_stream:
            chunk_dict = dataclass_to_dict(chunk, exclude={"messages"})
            # Check if this chunk contains image results
            if chunk_dict.get("images"):
                final_result = chunk_dict
        if not final_result:
            raise HTTPException(status_code=500, detail="No final image result found.")
        
        # Shorten the final result by returning only the content and images
        short_result = {
            "content": final_result.get("content"),
            "images": final_result.get("images")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"agent": "Image Agent", "prompt": prompt, "result": short_result}



if __name__ == "__main__":
    import uvicorn
    # Replace 'main:app' with 'filename:app' if you name your file differently.
    uvicorn.run("agent-2:app", host="localhost", port=8000, reload=True)
