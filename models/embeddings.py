import os
import numpy as np
from typing import List
import google.generativeai as genai
from config.config import GOOGLE_API_KEY, EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("Google API key not found")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = EMBEDDING_MODEL
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Google"""
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating single embedding: {e}")
            return []

def get_embedding_model():
    """Factory function to get embedding model"""
    try:
        return EmbeddingModel()
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        return None
