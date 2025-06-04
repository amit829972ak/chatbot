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

# Custom CSS - Corrected and improved
def load_css():
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for consistent theming */
    :root {
        --primary-color: #2D3748;
        --secondary-color: #4A5568;
        --accent-color: #0EA5E9;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --background-color: #F8FAFC;
        --surface-color: #FFFFFF;
        --text-primary: #1A202C;
        --text-secondary: #64748B;
        --border-color: #E2E8F0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
    }
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: var(--background-color);
        color: var(--text-primary);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--primary-color);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1rem;
    }
    
    /* Sidebar headers */
    .sidebar h1, .sidebar h2, .sidebar h3 {
        color: white !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Sidebar text */
    .sidebar .stMarkdown, .sidebar .stText {
        color: white;
    }
    
    /* Chat message container */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: var(--surface-color);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        margin-bottom: 1rem;
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, var(--accent-color) 0%, #0284C7 100%);
        color: white;
        border-radius: var(--radius-xl) var(--radius-xl) var(--radius-sm) var(--radius-xl);
        padding: 0.875rem 1.25rem;
        margin: 1rem 0 1rem auto;
        max-width: 75%;
        width: fit-content;
        margin-left: auto;
        box-shadow: var(--shadow-md);
        font-weight: 500;
        line-height: 1.5;
        word-wrap: break-word;
        display: block;
        clear: both;
    }
    
    /* Bot message styling */
    .bot-message {
        background-color: var(--surface-color);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-xl) var(--radius-xl) var(--radius-xl) var(--radius-sm);
        padding: 0.875rem 1.25rem;
        margin: 1rem auto 1rem 0;
        max-width: 75%;
        width: fit-content;
        box-shadow: var(--shadow-sm);
        line-height: 1.6;
        word-wrap: break-word;
        display: block;
        clear: both;
    }
    
    /* Document item styling */
    .doc-item {
        background-color: var(--surface-color);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin: 0.75rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .doc-item:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
        border-color: var(--accent-color);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color) 0%, #0284C7 100%);
        color: white;
        border-radius: var(--radius-md);
        border: none;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }
    
    /* Primary button variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--success-color) 0%, #059669 100%);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    
    /* Header styling */
    h1 {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 2.25rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: var(--secondary-color);
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: var(--radius-md);
        border: 2px solid var(--border-color);
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        background-color: var(--surface-color);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        outline: none;
    }
    
    /* Select box styling */
    .stSelectbox > div > div > select {
        border-radius: var(--radius-md);
        border: 2px solid var(--border-color);
        padding: 0.75rem 1rem;
        background-color: var(--surface-color);
        color: var(--text-primary);
        font-size: 0.875rem;
    }
    
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        outline: none;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-md);
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
        background-color: var(--surface-color);
    }
    
    .stFileUploader > div:hover {
        border-color: var(--accent-color);
        background-color: rgba(14, 165, 233, 0.02);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        font-size: 0.875rem;
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Success/Info/Warning message styling */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--success-color);
        border-radius: var(--radius-md);
        color: var(--success-color);
        padding: 0.75rem 1rem;
    }
    
    .stInfo {
        background-color: rgba(14, 165, 233, 0.1);
        border: 1px solid var(--accent-color);
        border-radius: var(--radius-md);
        color: var(--accent-color);
        padding: 0.75rem 1rem;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid var(--warning-color);
        border-radius: var(--radius-md);
        color: var(--warning-color);
        padding: 0.75rem 1rem;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: var(--accent-color) transparent var(--accent-color) transparent;
    }
    
    /* Knowledge base panel styling */
    .knowledge-base-panel {
        background-color: var(--surface-color);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            max-width: 90%;
        }
        
        h1 {
            font-size: 1.75rem;
        }
        
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: var(--background-color);
        border-radius: var(--radius-sm);
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: var(--radius-sm);
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
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
    st.sidebar.title("🤖 RAG Chatbot")
    
st.sidebar.title("⚙️ Admin Panel")

# Authentication
if not st.session_state['authenticated']:
    login_form()
else:
    st.sidebar.success("✅ Logged in")
    if st.sidebar.button("🚪 Logout"):
        st.session_state['authenticated'] = False
        st.rerun()
    
    # Document Management Section
    st.sidebar.header("📚 Document Management")
    
    # Upload section
    handle_upload()
    
    # Delete section
    handle_delete()
    
    # Display document versioning information
    display_document_info()

# AI Model Settings
st.sidebar.header("🧠 AI Model Settings")

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
api_key = st.sidebar.text_input("🔑 Enter API Key", type="password", value=st.session_state.get("api_key", ""))
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
    st.sidebar.info(f"ℹ️ **{model_choice}**: {model_info[model_choice]}")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.title("💬 Company RAG Chatbot")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1rem; border-radius: 0.75rem; margin-bottom: 1.5rem; border: 1px solid #e2e8f0;">
        <p style="margin: 0; color: #64748b; font-size: 0.95rem; line-height: 1.6;">
            Ask questions about company documents and policies. 
            The system will search through your documents to find relevant information.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state['chat_history']:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    # Input for new question with options
    col_input, col_upload, col_web_search = st.columns([4, 0.5, 0.5])
    
    with col_input:
        query = st.text_input("💭 Ask your question:", key="query_input", placeholder="Type your question here...")
    
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
        with st.spinner("🔍 Searching the web..."):
            search_results = handle_web_search(query)
        
        # Add search results to chat history
        st.session_state['chat_history'].append({'role': 'assistant', 'content': search_results})
        
        # Refresh the page to show the updated chat
        st.rerun()
    
    # Send button
    if st.button("🚀 Send", key="send_button", type="primary") and query:
        # Add user message to chat history
        st.session_state['chat_history'].append({'role': 'user', 'content': query})
        
        # Get response from RAG system
        with st.spinner("🤔 Getting answer..."):
            response = handle_chat(query, internal_model_choice, api_key, uploaded_file)
        
        # Add bot response to chat history
        st.session_state['chat_history'].append({'role': 'assistant', 'content': response})
        
        # Refresh the page to show the updated chat
        st.rerun()

# Document list and information panel
with col2:
    st.markdown('<div class="knowledge-base-panel">', unsafe_allow_html=True)
    
    doc_image_path = "static/documents.svg"
    if os.path.exists(doc_image_path):
        st.image(doc_image_path, width=200)
    st.header("📖 Knowledge Base")
    
    # Always show document list for all users
    documents = list_documents()
    if documents:
        st.subheader("📑 Available Documents")
        
        # Initialize selected documents in session state if not present
        if 'selected_documents' not in st.session_state:
            st.session_state['selected_documents'] = []
        
        # Create a container for document selection
        doc_container = st.container()
        
        # Add select all option
        select_all = st.checkbox("✅ Select All Documents", 
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
                if st.checkbox(f"📄 {doc}", value=is_selected, key=f"doc_{doc}"):
                    if doc not in st.session_state['selected_documents']:
                        st.session_state['selected_documents'].append(doc)
                else:
                    if doc in st.session_state['selected_documents']:
                        st.session_state['selected_documents'].remove(doc)
        
        # Show which documents are selected
        if st.session_state['selected_documents']:
            st.success(f"✨ Selected {len(st.session_state['selected_documents'])} documents for search")
        else:
            st.info("ℹ️ No documents selected. All documents will be searched.")
    else:
        if st.session_state['authenticated']:
            st.info("📝 No documents uploaded yet. Use the sidebar to upload documents.")
        else:
            st.info("📝 No documents available. Please contact an administrator to upload documents.")
    
    # Show information about the system
    st.subheader("ℹ️ About the System")
    st.markdown("""
    <div style="font-size: 0.85rem; line-height: 1.6; color: #64748b;">
    
    **📋 Document Support:**
    - PDF, Word (DOCX), Text files
    - Excel spreadsheets (XLSX, XLS)  
    - PowerPoint presentations (PPTX, PPT)
    
    **⚡ Advanced Features:**
    - Semantic search with sentence transformers
    - Document versioning and change tracking
    - Multiple AI model support (OpenAI, Google, Anthropic)
    - Web search integration
    - FAQ fallback system
    
    **🔄 How it works:**
    1. Upload documents using the clean '+' button
    2. Advanced embeddings create semantic understanding
    3. Smart search finds relevant content across all formats
    4. AI generates comprehensive answers
    5. Version tracking maintains document history
    
    **🔍 Search Options:**
    - 🌐 Web Search - Click the globe icon
    - 📎 Instant document analysis - Upload and ask
    - 📚 Knowledge base search - Select specific documents
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
