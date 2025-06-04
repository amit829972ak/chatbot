import streamlit as st
from utils_vector_store import search_index, extract_text_from_file
from utils_llm import query_model
import json
import os

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
    try:
        # Handle uploaded file first if provided
        temp_content = None
        if uploaded_file is not None:
            try:
                # Save the uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Extract text from the uploaded file
                temp_content = extract_text_from_file(temp_path)
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                if temp_content:
                    # Process the uploaded file content directly
                    prompt = f"""
                    Content from uploaded document:
                    {temp_content}
                    
                    User question: {query}
                    
                    Based only on the content from the uploaded document above, answer the user's question.
                    If the answer cannot be directly found in the document content, state that clearly.
                    Begin your response with: "Based on the uploaded document:"
                    """
                    
                    return query_model(prompt, model_choice, api_key)
                else:
                    return "I couldn't extract text from the uploaded file. Please make sure it's a valid PDF, DOCX, or TXT file."
                    
            except Exception as e:
                return f"Error processing uploaded file: {str(e)}"
        
        # Get selected documents from session state
        selected_docs = st.session_state.get('selected_documents', [])
        
        # First try to find relevant document chunks
        if selected_docs:
            # Search only in selected documents
            doc_paths = [os.path.join("uploaded_docs", doc) for doc in selected_docs]
            matches = search_index(query, specific_docs=doc_paths)
            doc_msg = f"Searched within {len(selected_docs)} selected documents"
        else:
            # Search all documents
            matches = search_index(query)
            doc_msg = "Searched all available documents"
        
        if matches and len(matches) > 0:
            # If we have relevant document matches, use them as context
            context = "\n\n".join([f"Document content: {match}" for match in matches])
            
            prompt = f"""
            Context information from company documents:
            {context}
            
            User question: {query}
            
            Based only on the context information provided, answer the user's question.
            If the answer cannot be directly found in the context, state that clearly.
            Begin your response with: "{doc_msg}"
            """
            
            return query_model(prompt, model_choice, api_key)
        else:
            # If no document matches, try FAQ
            faq_path = "faq.json"
            
            if os.path.exists(faq_path):
                with open(faq_path, "r") as f:
                    faq = json.load(f)
                
                # Look for matching FAQ questions
                for q, a in faq.items():
                    if q.lower() in query.lower() or query.lower() in q.lower():
                        return f"From FAQ:\n{a}"
            
            # If no FAQ matches either, let the AI try to answer generally
            prompt = f"""
            User question: {query}
            
            The user is asking about company information, but I couldn't find specific 
            documents or FAQ entries related to this question. 
            {doc_msg}, but no relevant information was found.
            
            Please provide a general response stating that you don't have specific 
            information on this topic and suggest what the user might do next.
            Begin your response with: "{doc_msg}, but no relevant information was found."
            """
            
            return query_model(prompt, model_choice, api_key)
    
    except Exception as e:
        return f"Sorry, I encountered an error while processing your request: {str(e)}"
