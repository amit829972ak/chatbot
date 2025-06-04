import streamlit as st
import tempfile
import os
from utils.vector_store import search_index, update_index_from_file, extract_text_from_file
from utils.llm import query_model
import json

# Load FAQ data
def load_faq():
    """Load FAQ data from JSON file."""
    try:
        with open("faq.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"faqs": []}
    except Exception as e:
        print(f"Error loading FAQ: {e}")
        return {"faqs": []}

def search_faq(query):
    """Search FAQ for relevant answers."""
    faq_data = load_faq()
    query_lower = query.lower()
    
    for faq in faq_data.get("faqs", []):
        question = faq.get("question", "").lower()
        keywords = [kw.lower() for kw in faq.get("keywords", [])]
        
        # Check if query matches question or keywords
        if any(keyword in query_lower for keyword in keywords) or any(word in question for word in query_lower.split()):
            return faq.get("answer", "")
    
    return None

def handle_chat(query, model_choice, api_key, uploaded_file=None):
    """
    Handle a chat query using RAG methodology.
    
    Args:
        query: User query string
        model_choice: Selected AI model
        api_key: API key for the selected model
        uploaded_file: Optional uploaded file to process and query
        
    Returns:
        Response string from the AI model
    """
    if not query.strip():
        return "Please enter a question."
    
    # Handle uploaded file if provided
    context_from_file = ""
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            # Extract text from the uploaded file
            file_content = extract_text_from_file(tmp_file_path)
            
            if file_content:
                context_from_file = f"\n\nContent from uploaded file '{uploaded_file.name}':\n{file_content[:3000]}..."
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            context_from_file = f"\n\nError processing uploaded file: {str(e)}"
    
    # Search for relevant documents in the knowledge base
    context_chunks = []
    
    # Check if specific documents are selected
    selected_docs = st.session_state.get('selected_documents', [])
    
    if selected_docs:
        # Search only in selected documents
        context_chunks = search_index(query, top_k=5, specific_docs=selected_docs)
    else:
        # Search all documents
        context_chunks = search_index(query, top_k=5)
    
    # Search FAQ first
    faq_answer = search_faq(query)
    if faq_answer:
        return f"**FAQ Answer:**\n\n{faq_answer}"
    
    # Build context for the AI model
    if context_chunks or context_from_file:
        context = "Based on the following information, please provide a comprehensive answer:\n\n"
        
        if context_chunks:
            context += "**Relevant document sections:**\n"
            for i, chunk in enumerate(context_chunks, 1):
                context += f"\n{i}. {chunk}\n"
        
        if context_from_file:
            context += context_from_file
        
        context += f"\n\n**Question:** {query}\n\n"
        context += "Please provide a detailed answer based on the above information. If the information doesn't fully address the question, clearly state what aspects cannot be answered from the provided context."
        
        # Query the AI model
        response = query_model(context, model_choice, api_key)
        return response
    
    else:
        # No relevant context found
        if not api_key:
            return "No relevant documents found for your query. Please provide an API key to get a general response, or upload documents to search through."
        
        # Provide general response without document context
        general_prompt = f"Please provide a helpful response to this question: {query}\n\nNote: This response is not based on any specific documents from the knowledge base."
        response = query_model(general_prompt, model_choice, api_key)
        
        return f"**General Response** (No specific documents found):\n\n{response}"
