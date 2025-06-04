import streamlit as st
import base64
from utils.auth import login_form
from utils.docs import handle_upload, handle_delete, list_documents, display_document_info
from utils.chat import handle_chat
import os

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ""
if 'model_choice' not in st.session_state:
    st.session_state['model_choice'] = "OpenAI GPT"

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    /* Primary colors */
    :root {
        --primary: #2D3748;
        --secondary: #4A5568;
        --accent: #0EA5E9;
        --background: #F7FAFC;
        --text: #1A202C;
    }
    
    /* Font styling */
    body {
        font-family: 'Inter', 'SF Pro Display', sans-serif;
        color: var(--text);
        background-color: var(--background);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--primary);
        color: white;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: var(--accent);
        color: white;
        border-radius: 15px 15px 0 15px;
        padding: 10px 16px;
        margin: 16px 0;
        max-width: 80%;
        align-self: flex-end;
    }
    
    .bot-message {
        background-color: #f0f0f0;
        color: var(--text);
        border-radius: 15px 15px 15px 0;
        padding: 10px 16px;
        margin: 16px 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Document list styling */
    .doc-item {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .doc-item:hover {
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--accent);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #0b8bcf;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--primary);
        font-weight: 600;
    }
    
    /* Input styling */
    input, textarea, select {
        border-radius: 6px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_css()

# Create uploaded_docs directory if it doesn't exist
os.makedirs("uploaded_docs", exist_ok=True)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Sidebar configuration
sidebar_logo_path = "static/logo.svg"
if os.path.exists(sidebar_logo_path):
    st.sidebar.image(sidebar_logo_path, width=250)
else:
    st.sidebar.title("RAG Chatbot")
    
st.sidebar.title("Admin Panel")

# Authentication
if not st.session_state['authenticated']:
    login_form()
else:
    st.sidebar.success("Logged in ✅")
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.rerun()
    
    # Document Management Section
    st.sidebar.header("Document Management")
    
    # Upload section
    handle_upload()
    
    # Delete section
    handle_delete()
    
    # Display document versioning information
    display_document_info()

# AI Model Settings
st.sidebar.header("AI Model Settings")

# Define model options with versions
openai_models = [
    "OpenAI GPT-4o",
    "OpenAI GPT-4",
    "OpenAI GPT-3.5 Turbo"
]
gemini_models = [
    "Google Gemini Pro",
    "Google Gemini Flash",
    "Google Gemini 1.0 Pro Vision",
    "Google Gemini 1.5 Pro",
    "Google Gemini 1.5 Flash",
    "Google Gemini 1.5 Pro Latest",
    "Google Gemini 1.5 Flash Latest",
    "Google Gemini 2.0 Pro Vision",
    "Google Gemini 2.0 Pro",
    "Google Gemini 2.5 Pro",
    "Google Gemini 2.5 Flash"
]
claude_models = [
    "Claude 3.5 Sonnet",
    "Claude 3 Opus",
    "Claude 3 Sonnet",
    "Claude 3 Haiku"
]

# Create expanded model options
model_categories = {
    "OpenAI Models": openai_models,
    "Google Models": gemini_models,
    "Anthropic Models": claude_models
}

# Select model category first
model_category = st.sidebar.selectbox(
    "Select Model Provider",
    list(model_categories.keys())
)

# Then select specific model from that category
model_choice = st.sidebar.selectbox(
    "Select Specific Model",
    model_categories[model_category]
)

# Map from display name to internal name for code use
if "OpenAI" in model_choice:
    internal_model_choice = "OpenAI GPT"
elif "Google" in model_choice:
    internal_model_choice = "Google Gemini"
elif "Claude" in model_choice:
    internal_model_choice = "Claude"
else:
    internal_model_choice = "OpenAI GPT"  # Default

# Update session state with internal model choice and full model name
st.session_state["model_choice"] = internal_model_choice
st.session_state["specific_model"] = model_choice

# API key input
api_key = st.sidebar.text_input("Enter API Key", type="password", value=st.session_state.get("api_key", ""))
st.session_state["api_key"] = api_key

# Model information
model_info = {
    "OpenAI GPT-4o": "OpenAI's most advanced multimodal model with vision capabilities (Mar 2024)",
    "OpenAI GPT-4": "OpenAI's powerful language model (Mar 2023)",
    "OpenAI GPT-3.5 Turbo": "OpenAI's efficient and cost-effective model (Jan 2023)",
    
    "Google Gemini Pro": "Google's advanced reasoning model (Dec 2023)",
    "Google Gemini Flash": "Google's fastest model for efficiency (Dec 2023)",
    "Google Gemini 1.0 Pro Vision": "Google's first multimodal model with vision capabilities (Dec 2023)",
    "Google Gemini 1.5 Pro": "Google's enhanced reasoning model (Mar 2024)",
    "Google Gemini 1.5 Flash": "Google's enhanced fast model (Mar 2024)",
    "Google Gemini 1.5 Pro Latest": "Latest version of Google's Gemini 1.5 Pro model (May 2024)",
    "Google Gemini 1.5 Flash Latest": "Latest version of Google's Gemini 1.5 Flash model (May 2024)",
    "Google Gemini 2.0 Pro Vision": "Google's advanced multimodal model with vision capabilities (Feb 2025)",
    "Google Gemini 2.0 Pro": "Google's powerful second-generation model (Feb 2025)",
    "Google Gemini 2.5 Pro": "Google's cutting-edge Pro model (Apr 2025)",
    "Google Gemini 2.5 Flash": "Google's cutting-edge fast model (Apr 2025)",
    
    "Claude 3.5 Sonnet": "Anthropic's latest model with advanced capabilities (Oct 2024)",
    "Claude 3 Opus": "Anthropic's most powerful model (Mar 2024)",
    "Claude 3 Sonnet": "Anthropic's balanced model (Mar 2024)",
    "Claude 3 Haiku": "Anthropic's fastest model (Mar 2024)"
}

# Show model information
if model_choice in model_info:
    st.sidebar.info(f"Using {model_choice}: {model_info[model_choice]}")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Company RAG Chatbot")
    st.markdown("""
    Ask questions about company documents and policies. 
    The system will search through your documents to find relevant information.
    """)
    
    # Display chat history
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Input for new question with options
    col_input, col_upload, col_web_search = st.columns([4, 0.5, 0.5])
    
    with col_input:
        query = st.text_input("Ask your question:", key="query_input")
    
    with col_upload:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some space to align with text input
        uploaded_file = st.file_uploader(
            "📎", 
            type=["pdf", "txt", "docx", "xlsx", "xls", "pptx", "ppt"],
            help="Upload a document to ask questions about (PDF, TXT, DOCX, XLSX, PPTX)",
            label_visibility="collapsed",
            key="doc_uploader_inline"
        )
    
    with col_web_search:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some space to align with text input
        web_search_clicked = st.button("🌐", help="Search the web for this question", key="web_search_btn")
    
    # Handle web search
    if web_search_clicked and query:
        from utils.web_search import handle_web_search
        
        # Add user message to chat history
        st.session_state['chat_history'].append({'role': 'user', 'content': f"🌐 Web Search: {query}"})
        
        # Perform web search
        with st.spinner("Searching the web..."):
            search_results = handle_web_search(query)
        
        # Add search results to chat history
        st.session_state['chat_history'].append({'role': 'assistant', 'content': search_results})
        
        # Refresh the page to show the updated chat
        st.rerun()
    
    if st.button("Send") and query:
        # Add user message to chat history
        st.session_state['chat_history'].append({'role': 'user', 'content': query})
        
        # Get response from RAG system
        with st.spinner("Getting answer..."):
            response = handle_chat(query, internal_model_choice, api_key, uploaded_file)
        
        # Add bot response to chat history
        st.session_state['chat_history'].append({'role': 'assistant', 'content': response})
        
        # Refresh the page to show the updated chat
        st.rerun()

# Document list and information panel
with col2:
    doc_image_path = "static/documents.svg"
    if os.path.exists(doc_image_path):
        st.image(doc_image_path, width=250)
    st.header("Knowledge Base")
    
    # Always show document list for all users
    documents = list_documents()
    if documents:
        st.subheader("Available Documents")
        
        # Initialize selected documents in session state if not present
        if 'selected_documents' not in st.session_state:
            st.session_state['selected_documents'] = []
        
        # Create a container for document selection
        doc_container = st.container()
        
        # Add select all option
        select_all = st.checkbox("Select All Documents", 
                                key="select_all_docs",
                                value=len(st.session_state['selected_documents']) == len(documents))
        
        if select_all:
            st.session_state['selected_documents'] = documents
        elif select_all == False and len(st.session_state['selected_documents']) == len(documents):
            st.session_state['selected_documents'] = []
        
        # Display document list with checkboxes
        with doc_container:
            for doc in documents:
                is_selected = doc in st.session_state['selected_documents']
                if st.checkbox(doc, value=is_selected, key=f"doc_{doc}"):
                    if doc not in st.session_state['selected_documents']:
                        st.session_state['selected_documents'].append(doc)
                else:
                    if doc in st.session_state['selected_documents']:
                        st.session_state['selected_documents'].remove(doc)
        
        # Show which documents are selected
        if st.session_state['selected_documents']:
            st.success(f"Selected {len(st.session_state['selected_documents'])} documents for search")
        else:
            st.info("No documents selected. All documents will be searched.")
    else:
        if st.session_state['authenticated']:
            st.info("No documents uploaded yet. Use the sidebar to upload documents.")
        else:
            st.info("No documents available. Please contact an administrator to upload documents.")
    
    # Show information about the system
    st.subheader("About the System")
    st.markdown("""
    This advanced RAG (Retrieval-Augmented Generation) chatbot features:
    
    **Document Support:**
    - PDF, Word (DOCX), Text files
    - Excel spreadsheets (XLSX, XLS)
    - PowerPoint presentations (PPTX, PPT)
    
    **Advanced Features:**
    - Semantic search with sentence transformers
    - Document versioning and change tracking
    - Multiple AI model support (OpenAI, Google, Anthropic)
    - Web search integration
    - FAQ fallback system
    
    **How it works:**
    1. Upload documents using the clean '+' button
    2. Advanced embeddings create semantic understanding
    3. Smart search finds relevant content across all formats
    4. AI generates comprehensive answers
    5. Version tracking maintains document history
    
    **Search Options:**
    - 🌐 Web Search - Click the globe icon
    - 📎 Instant document analysis - Upload and ask
    - 📚 Knowledge base search - Select specific documents
    """)
