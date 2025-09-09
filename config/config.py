# config/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Model Configuration
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Directories
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")

# Response modes
RESPONSE_MODES = {
    "concise": "Provide brief, concise answers. Keep responses short and to the point.",
    "detailed": "Provide detailed, comprehensive answers with explanations and examples when helpful."
}

def get_system_prompt(response_mode="detailed", context=""):
    """Get system prompt based on response mode and context"""
    base_prompt = "You are a helpful AI assistant."
    
    if response_mode.lower() in RESPONSE_MODES:
        mode_instruction = RESPONSE_MODES[response_mode.lower()]
    else:
        mode_instruction = RESPONSE_MODES["detailed"]
    
    prompt = f"{base_prompt} {mode_instruction}"
    
    if context:
        prompt += f"\n\nContext from knowledge base:\n{context}"
    
    return prompt
