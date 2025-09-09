import streamlit as st
import os
from models.llm import get_chat_model, get_response
from utils.document_processor import process_uploaded_files
from utils.rag_system import SimpleRAG
from utils.web_search import search_web, search_web_duckduckgo

# Page config
st.set_page_config(
    page_title="NeoStats AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Add this check at the beginning of main()
def check_api_keys():
    """Check if required API keys are available"""
    from config.config import GROQ_API_KEY, GOOGLE_API_KEY
    
    missing_keys = []
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    
    if missing_keys:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.error("Please check your .env file")
        return False
    return True

def main():
    if not check_api_keys():
        return
        
    st.title("🤖 NeoStats AI Chatbot with RAG & Web Search")
    # ... rest of your existing main() function
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SimpleRAG()
        st.session_state.rag_system.load_vectorstore()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Response mode
        response_mode = st.selectbox(
            "Response Mode:",
            ["Concise", "Detailed"]
        )
        
        # Web search toggle
        enable_web_search = st.checkbox("Enable Web Search", value=True)
        
        st.divider()
        
        # Document upload
        st.subheader("📚 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    chunks = process_uploaded_files(uploaded_files)
                    if chunks:
                        success = st.session_state.rag_system.add_documents(chunks)
                        if success:
                            st.success(f"Added {len(chunks)} document chunks!")
                        else:
                            st.error("Failed to process documents")
                    else:
                        st.error("No text extracted from files")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        # Show knowledge base status
        num_docs = len(st.session_state.rag_system.documents)
        if num_docs > 0:
            st.info(f"Knowledge base: {num_docs} chunks loaded")
        
        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize client
                    client = get_chat_model()
                    
                    if not client:
                        st.error("Failed to initialize chat model. Please check your API key.")
                        return
                    
                    # Search for relevant documents
                    relevant_docs = st.session_state.rag_system.search_similar(prompt, k=3)
                    
                    # Build context
                    context = ""
                    sources_used = []
                    
                    if relevant_docs:
                        context += "\n\nRelevant information from knowledge base:\n"
                        for i, doc in enumerate(relevant_docs):
                            context += f"- {doc['text'][:300]}...\n"
                        sources_used.append("Knowledge Base")
                    
                    # Add web search if enabled and no relevant docs found or user asks for current info
                    if enable_web_search and (not relevant_docs or any(word in prompt.lower() for word in ['latest', 'recent', 'current', 'today', 'news', '2024', '2025'])):
                        web_results = search_web(prompt, num_results=2)
                        
                        # Fallback to DuckDuckGo if SerpAPI fails
                        if not web_results:
                            web_results = search_web_duckduckgo(prompt, num_results=2)
                        
                        if web_results:
                            context += "\n\nWeb search results:\n"
                            for result in web_results:
                                context += f"- {result['title']}: {result['snippet']}\n"
                            sources_used.append("Web Search")
                    
                    # Build system message based on response mode
                    if response_mode == "Concise":
                        system_msg = "You are a helpful AI assistant. Provide concise, brief answers. Keep responses short and to the point."
                    else:
                        system_msg = "You are a helpful AI assistant. Provide detailed, comprehensive answers with explanations and examples when helpful."
                    
                    if context:
                        system_msg += context
                    
                    # Prepare messages
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Get response
                    response = get_response(client, messages)
                    
                    # Add sources information
                    if sources_used:
                        response += f"\n\n*Sources: {', '.join(sources_used)}*"
                    
                    st.write(response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Make sure your OpenAI API key is set correctly in the .env file")

if __name__ == "__main__":
    main()
