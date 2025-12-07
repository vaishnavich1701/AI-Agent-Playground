# agent_creation_playground_with_dynamic_endpoint.py

import io
import sys
import re
import json
import os
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Import AGNO core classes
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Import only the installed AGNO tools:
from agno.tools.duckduckgo import DuckDuckGoTools            # Enables web search using DuckDuckGo.
from agno.tools.yfinance import YFinanceTools                  # Retrieves financial data from Yahoo Finance.

# Import OpenAI package for GPT calls
import openai
from fastapi import Query

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

##################################################
# Clean Response Function (as provided)         #
##################################################
def clean_response(response: str) -> str:
    """
    Cleans the agent response by:
    - Removing ANSI escape sequences.
    - Splitting the response into lines.
    - Skipping lines that are entirely decorative.
    - Stripping any border characters (┏, ┗, ┃, ━, or ┛) from the start and end of each line.
    - Removing any stray decorative characters (e.g. "┛") in the final text.
    - Joining the remaining lines with newlines to preserve paragraph breaks.
    """
    response = re.sub(r'\x1b\[[0-9;]*m', '', response)
    lines = response.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or all(ch in "┏┗┃━┛" for ch in stripped):
            continue
        line = re.sub(r'^[┏┗┃━┛\s]+', '', line)
        line = re.sub(r'[┏┗┃━┛\s]+$', '', line)
        if re.match(r'^(Message|Response)', line):
            continue
        line = re.sub(r'[ ]+', ' ', line).strip()
        cleaned_lines.append(line)
    cleaned_response = "\n".join(cleaned_lines)
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
        show_tool_calls=False,
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
            show_tool_calls=False,
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

##################################################
# Main Application Runner                        #
##################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9000, reload=True)
