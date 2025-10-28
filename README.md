# AI-Powered Agent Assist with Seamless AI-Human Handoff

AI-powered chatbot designed to handle telecom support queries, detect intent and sentiment, and seamlessly hand off conversations to human agents when needed.

## Key Features

1. __Intent + Sentiment Detection__ – Powered by Cohere LLM  
2. __Contextual Memory__ – Uses Qdrant vector search for past conversation retrieval  
3. __Seamless AI-Human Handoff__ – Automatically passes context, intent, and sentiment to the agent  
4. __Agent Assist Dashboard__ – AI co-pilot suggests replies and summarizes interactions  
5. __Streamlit UI__ – Simple, interactive chat interface  
6. __Scalable, Modular Design__ – Plug-and-play with different LLMs or databases  

## System Architecture

```

User → Chatbot / Voicebot
↓
Intent & Sentiment Analyzer (Cohere)
↓
Context Memory (Qdrant)
↓
AI Response Generator
↓
Escalation & Handoff Engine
↓
Human Agent Assist Dashboard
```

## Tech Stack
Layer	Technology  
Language Model	Cohere API  
Vector Database	Qdrant  
Backend / Orchestration	Python 3.10, dotenv  
Frontend	Streamlit  
Memory & Context	LLM Embeddings + Similarity Search  
Deployment	GitHub Codespaces / Cloud-ready  

## Setup & Installation

All environment setup steps are automated in the provided script : __bash setup_chatbot_env.sh__

### Note:
This script creates the Conda environment, installs all dependencies (Cohere, Qdrant, Streamlit, etc.), and registers a Jupyter kernel.
You only need to run it once before launching the app.

## Environment Variables

Create a .env file in the project root (copy from .env.example):

CO_API_KEY=<your_cohere_api_key>  
QDRANT_URL=<https://your-qdrant-instance-url>  
QDRANT_API_KEY=<your_qdrant_api_key>  

## Files

__app.py__ -  Frontend code  
__llm_pipeline.py__ - Contains Retrieval, Augmentation and generation   

## Running the App

After setup, launch the chatbot interface with : 
__streamlit run app.py__

App will be available at:
http://localhost:8501
or via the “Preview Port” in GitHub Codespaces.

### __Developed for: Tech Odyssey Hackathon 2025__
