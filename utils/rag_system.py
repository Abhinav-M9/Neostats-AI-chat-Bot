import os
import pickle
import numpy as np
from typing import List, Dict, Optional
import faiss
from models.embeddings import get_embedding_model
from config.config import VECTORSTORE_DIR

class SimpleRAG:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.documents = []
        self.embeddings = []
        self.index = None
        self.vectorstore_path = os.path.join(VECTORSTORE_DIR, "vectorstore.pkl")
        self.index_path = os.path.join(VECTORSTORE_DIR, "faiss_index")
        
        # Create vectorstore directory if it doesn't exist
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    def add_documents(self, documents: List[str]) -> bool:
        """Add documents to the RAG system"""
        if not self.embedding_model:
            print("Embedding model not available")
            return False
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.get_embeddings(documents)
            if not embeddings:
                print("Failed to generate embeddings")
                return False
            
            # Add to storage
            self.documents.extend(documents)
            self.embeddings.extend(embeddings)
            
            # Update FAISS index
            self._update_index()
            
            # Save to disk
            self._save_vectorstore()
            
            print(f"Successfully added {len(documents)} documents")
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def _update_index(self):
        """Update FAISS index with current embeddings"""
        if not self.embeddings:
            return
        
        try:
            embeddings_array = np.array(self.embeddings).astype('float32')
            dimension = embeddings_array.shape[1]
            
            # Create new index
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_array)
            print(f"Updated FAISS index with {len(self.embeddings)} embeddings")
        except Exception as e:
            print(f"Error updating index: {e}")
    
    def search_similar(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        if not self.embedding_model or not self.index or not self.documents:
            print("RAG system not properly initialized")
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.get_single_embedding(query)
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Search
            query_vector = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_vector, min(k, len(self.documents)))
            
            # Return results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'score': float(distance),
                        'index': int(idx)
                    })
            
            print(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            print(f"Error searching similar documents: {e}")
            return []
    
    def _save_vectorstore(self):
        """Save vectorstore to disk"""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings
            }
            with open(self.vectorstore_path, 'wb') as f:
                pickle.dump(data, f)
            
            if self.index:
                faiss.write_index(self.index, self.index_path)
            
            print("Vectorstore saved successfully")
        except Exception as e:
            print(f"Error saving vectorstore: {e}")
    
    def load_vectorstore(self):
        """Load vectorstore from disk"""
        try:
            if os.path.exists(self.vectorstore_path):
                with open(self.vectorstore_path, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.embeddings = data.get('embeddings', [])
                
                if os.path.exists(self.index_path) and self.embeddings:
                    self.index = faiss.read_index(self.index_path)
                elif self.embeddings:
                    self._update_index()
                    
                print(f"Loaded {len(self.documents)} documents from vectorstore")
            else:
                print("No existing vectorstore found")
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
