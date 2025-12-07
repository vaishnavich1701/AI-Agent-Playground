from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware 
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import io
import sys
import re
import uuid
from typing import Any, Dict
import json
from typing import Iterator
from agno.tools.dalle import DalleTools
from agno.utils.common import dataclass_to_dict
import openai
from fastapi import Query
import os
import dotenv
dotenv.load_dotenv()
# Set your OpenAI API key and MongoDB connection string (ensure they are set in your environment)
openai.api_key = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_CONNECTION_STRING")

# Initialize MongoDB client using Motor (async) and create a new collection "custom_agents"
from motor.motor_asyncio import AsyncIOMotorClient
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["agent_playground_db"]
agents_collection = db["custom_agents"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# News Agent with DuckDuckGo tool 
news_agent = Agent(
    name="News Agent",
    description=(
        "You are an enthusiastic news reporter with a flair for storytelling!",
        "Return your output in plain Markdown format without any decorative border characters (such as ┏, ┗, ┃, ━, or ┛)."
    ),
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    markdown=False,
)

# Finance Agent with Yfinance tool
finance_agent = Agent(
    name="Finance Agent",
    description=(
        "You are an expert finance analyst providing detailed stock insights.",
        "Return your output in plain Markdown format."
    ),
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools()],  # <-- Remove arguments
    markdown=False,
)



# Image agent using Dall-e 
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

# Travel Planner agent with DuckDuckGo for Itinerary
travel_agent = Agent(
    name="Travel Planner Agent",
    description=(
        "You are a travel planner that provides comprehensive itineraries, hotel recommendations, local attractions, and travel tips for any destination.",
        "Return your output in plain Markdown format without any decorative border characters (such as ┏, ┗, ┃, ━, or ┛)."
    ),
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    markdown=False,
    instructions=[
        "Plan a detailed travel itinerary for the given destination.",
        "Include recommendations for attractions, accommodations, dining, and transportation.",
        "Return the itinerary in a clear markdown format with headings for each day or section.",
    ]
)

def clean_response(response: str) -> str:
    """
    Cleans the agent response by:
    - Removing ANSI escape sequences.
    - Splitting the response into lines.
    - Skipping lines that are entirely decorative.
    - Stripping any border characters (┏, ┗, ┃, ━, ┛) from the start and end of each line.
    - Removing any stray decorative characters (e.g. "┛") in the final text.
    - Joining the remaining lines with newlines to preserve paragraph breaks.
    """
    # Remove ANSI escape sequences
    response = re.sub(r'\x1b\[[0-9;]*m', '', response)
    
    # Split into lines
    lines = response.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines or lines that are fully decorative
        if not stripped or all(ch in "┏┗┃━┛" for ch in stripped):
            continue
        # Remove border characters from the beginning and end of the line
        line = re.sub(r'^[┏┗┃━┛\s]+', '', line)
        line = re.sub(r'[┏┗┃━┛\s]+$', '', line)
        # Also skip lines that start with "Message" or "Response" (decorative headings)
        if re.match(r'^(Message|Response)', line):
            continue
        # Normalize spaces within the line (but keep newlines intact later)
        line = re.sub(r'[ ]+', ' ', line).strip()
        cleaned_lines.append(line)
    
    # Join the cleaned lines with newline characters to preserve Markdown formatting
    cleaned_response = "\n".join(cleaned_lines)
    
    # Remove any stray "┛" characters
    cleaned_response = cleaned_response.replace("┛", "")
    
    return cleaned_response


##################################################
# GPT Layer to Generate Agent Parameters         #
##################################################
def call_openai_api(messages: Any) -> str:
    """
    Calls OpenAI's ChatCompletion API with the given messages and returns the assistant's response.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # or "gpt-4o" as applicable
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")

def generate_agent_params_gpt(description: str) -> Dict[str, str]:
    """
    Uses GPT to generate a JSON object with:
      - agent_name
      - description_gpt
      - tool_gpt
    Agent type is inferred automatically from description.
    """
    tools_context = """
List of available AGNO tools and their functions:
- DuckDuckGoTools(): Enables web search using DuckDuckGo.
- YFinanceTools(): Retrieves financial data from Yahoo Finance.
"""
    system_prompt = f"""
You are an expert in AGNO agent creation. You have full knowledge of the available AGNO tools.
{tools_context}
Your job is to:
  1. Understand the user's intent and infer the agent type from the description.
  2. Generate a JSON object with exactly the following keys:
     - "agent_name": A concise name for the agent.
     - "description_gpt": A detailed description/instructions for the agent.
     - "tool_gpt": The best suited AGNO tool to be used (e.g., DuckDuckGoTools(), YFinanceTools()).

The agent will eventually be instantiated using the following:
   agent = Agent(
        name="agent_name",
        description=description_gpt,
        model=OpenAIChat(id="gpt-4o"),
        tools=[tool_gpt],
        markdown=False,
   )

Do not include extra commentary—just return valid JSON.

DO NOT include anything other than valid JSON in your response.
DO NOT even add leading or trailing characters like "'''" or "'''json", etc.

User description:
\"\"\"{description}\"\"\"
"""
    messages = [{"role": "system", "content": system_prompt}]
    gpt_response = call_openai_api(messages)
    try:
        params = json.loads(gpt_response)
        if not all(key in params for key in ["agent_name", "description_gpt", "tool_gpt"]):
            raise ValueError("Missing required keys in GPT response.")
        return params
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse GPT response as JSON: {e}")


##################################################
# Agent Creation Playground Endpoint             #
##################################################
@app.post("/create_agent")
async def create_agent(
    description: str = Query(..., description="Describe what your agent should do")
):
    """
    Creates an agent by inferring the type from description.
    """
    try:
        params = generate_agent_params_gpt(description)

        unique_endpoint = f"agent_{uuid.uuid4().hex[:8]}"

        document = {
            "agent_name": params["agent_name"],
            "description_gpt": params["description_gpt"],
            "tool_gpt": params["tool_gpt"],
            "endpoint": unique_endpoint
        }
        result = await agents_collection.insert_one(document)

        agent_url = f"http://localhost:9000/agent/{unique_endpoint}"
        return {
            "status": "success",
            "agent_params": params,
            "endpoint": unique_endpoint,
            "agent_url": agent_url,
            "mongo_id": str(result.inserted_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##################################################
# Dynamic Agent Query Endpoint                   #
##################################################
@app.get("/agent/{custom_endpoint}")
async def run_custom_agent(custom_endpoint: str, query: str):
    """
    Dynamic endpoint to run a custom agent.
    Retrieves the agent parameters from MongoDB based on the unique endpoint,
    instantiates the agent, and returns the agent's response to the query.
    
    The query parameter is provided as a query string in the URL.
    For example: GET /agent/agent_ab12cd34?query=latest+news
    """
    try:
        doc = await agents_collection.find_one({"endpoint": custom_endpoint})
        if not doc:
            raise HTTPException(status_code=404, detail="Agent not found for this endpoint.")
        
        # Dynamically instantiate the tool (ensure this is used in a trusted environment).
        tool_instance = eval(doc["tool_gpt"])
        
        agent_instance = Agent(
            name=doc["agent_name"],
            description=doc["description_gpt"],
            model=OpenAIChat(id="gpt-4o"),
            tools=[tool_instance],
            markdown=False
        )
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        agent_instance.print_response(query, stream=False)
        response = mystdout.getvalue()
        sys.stdout = old_stdout
        
        cleaned_response = clean_response(response)
        return {"agent": f"{doc['agent_name']} Agent", "query": query, "response": cleaned_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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

@app.get("/travel")
async def get_travel(query: str):
    """
    Endpoint to get a travel itinerary for the given destination or travel query.
    Example: GET /travel?query=3-day+itinerary+in+Paris
    """
    try:
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        travel_agent.print_response(query, stream=False)
        response = mystdout.getvalue()
        sys.stdout = old_stdout
        cleaned_response = clean_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"agent": "Travel Planner Agent", "query": query, "response": cleaned_response}



if __name__ == "__main__":
    import uvicorn
    # Replace 'main:app' with 'filename:app' if you name your file differently.
    uvicorn.run("agent-3:app", host="localhost", port=8000, reload=True)