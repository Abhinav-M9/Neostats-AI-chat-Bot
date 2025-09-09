import os
from typing import Dict, List
from config.config import GROQ_API_KEY, GROQ_MODEL
from groq import Groq

def get_chat_model():
    """Initialize and return chat model using Groq"""
    if not GROQ_API_KEY:
        print("Error: Missing Groq API key")
        return None
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return {"provider": "groq", "client": client, "model": GROQ_MODEL}
    except Exception as e:
        print(f"Groq initialization failed: {e}")
        return None

def get_response(chat_model_dict: Dict, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """Get response from the chat model"""
    
    if not chat_model_dict:
        return "Error: No chat model available. Please check your API key."
    
    provider = chat_model_dict["provider"]
    client = chat_model_dict["client"]
    model = chat_model_dict["model"]
    
    try:
        if provider == "groq":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,  # Increased token limit
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content

    except Exception as e:
        # More detailed error handling
        if "model_decommissioned" in str(e):
            return f"Error: The model '{model}' has been decommissioned. Please update your GROQ_MODEL in the .env file to one of these current models: llama-3.3-70b-versatile, llama-3.1-8b-instant, or gemma2-9b-it"
        else:
            return f"Error getting response: {str(e)}"
    
    return "Error: Unsupported provider"
