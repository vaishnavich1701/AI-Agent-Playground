#  AI-Agent-Playground

Welcome to the backend engine powering the Agent ecosystem — a lightweight Python-based AI agent designed for real-time processing and automation using OpenAI's API.

---

## Getting Started

Follow these steps to set up and run the backend locally:

### 1. Clone the repository

### 2. Create a virtual environment
```bash
python3 -m venv venv
```
### 3. Activate the virtual environment
macOS / Linux:
```bash
source venv/bin/activate
```

Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set your OpenAI API key
macOS / Linux:
```bash
export OPENAI_API_KEY=your_openai_key_here
```

Windows (PowerShell):
```bash
$env:OPENAI_API_KEY = "your_openai_key_here"
```
### 6. Run the backend
```bash
python agent-1.py
```

## API Documentation
Once the server is running, navigate to:

```bash
http://localhost:8000/docs
```
This opens the Swagger UI, where you can view all available endpoints, test them interactively, and understand input/output formats.


