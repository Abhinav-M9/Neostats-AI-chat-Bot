import streamlit as st
import os
from models.llm import get_chat_model, get_response
from utils.document_processor import process_uploaded_files
from utils.rag_system import SimpleRAG
from utils.web_search import search_web, search_web_duckduckgo

st.set_page_config(
    page_title="NeoStats AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

def check_api_keys():
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
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SimpleRAG()
        st.session_state.rag_system.load_vectorstore()
    
    with st.sidebar:
        st.header("Configuration")
        
        response_mode = st.selectbox(
            "Response Mode:",
            ["Concise", "Detailed"]
        )
        
        enable_web_search = st.checkbox("Enable Web Search", value=True)
        
        st.divider()
        
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
        
        num_docs = len(st.session_state.rag_system.documents)
        if num_docs > 0:
            st.info(f"Knowledge base: {num_docs} chunks loaded")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = get_chat_model()
                    
                    if not client:
                        st.error("Failed to initialize chat model. Please check your API key.")
                        return
                    
                    relevant_docs = st.session_state.rag_system.search_similar(prompt, k=3)
                    
                    context = ""
                    sources_used = []
                    
                    if relevant_docs:
                        context += "\n\nRelevant information from knowledge base:\n"
                        for i, doc in enumerate(relevant_docs):
                            context += f"- {doc['text'][:300]}...\n"
                        sources_used.append("Knowledge Base")
                    
                    if enable_web_search and (not relevant_docs or any(word in prompt.lower() for word in ['latest', 'recent', 'current', 'today', 'news', '2024', '2025'])):
                        web_results = search_web(prompt, num_results=2)
                        
                        if not web_results:
                            web_results = search_web_duckduckgo(prompt, num_results=2)
                        
                        if web_results:
                            context += "\n\nWeb search results:\n"
                            for result in web_results:
                                context += f"- {result['title']}: {result['snippet']}\n"
                            sources_used.append("Web Search")
                    
                    if response_mode == "Concise":
                        system_msg = "You are a helpful AI assistant. Provide concise, brief answers. Keep responses short and to the point."
                    else:
                        system_msg = "You are a helpful AI assistant. Provide detailed, comprehensive answers with explanations and examples when helpful."
                    
                    if context:
                        system_msg += context
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = get_response(client, messages)
                    
                    if sources_used:
                        response += f"\n\n*Sources: {', '.join(sources_used)}*"
                    
                    st.write(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Make sure your OpenAI API key is set correctly in the .env file")

if __name__ == "__main__":
    main()
