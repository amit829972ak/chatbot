import streamlit as st
import time
import os
import json
import hashlib
import io
import faiss
import numpy as np
import pickle
import requests
from datetime import datetime
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL = "gpt-4o"

# Configuration parameters
VECTOR_DB_PATH = "vector_db"

# Department information
DEPARTMENTS = [
    "HR",
    "Finance",
    "Marketing",
    "IT",
    "Sales"
]

# User roles
USER_ROLES = [
    "Employee",
    "Manager",
    "Department Head",
    "Executive"
]

# Color scheme
COLORS = {
    "primary": "#0078D4",  # corporate blue
    "secondary": "#107C10",  # enterprise green
    "neutral": "#F5F5F5",  # light grey
    "text": "#252525",  # dark grey
    "accent": "#5C2D91",  # professional purple
}

# Department information (static for demo purposes)
DEPARTMENT_INFO = {
    "HR": {
        "description": "The Human Resources department handles employee-related matters including recruitment, onboarding, benefits, and workplace policies.",
        "key_contacts": {
            "director": "HR Director",
            "general_inquiries": "hr@company.com"
        },
        "common_topics": [
            "Employee benefits",
            "Leave policies",
            "Performance reviews",
            "Recruitment process",
            "Workplace harassment policies",
            "Employee training and development"
        ]
    },
    "Finance": {
        "description": "The Finance department oversees the company's financial operations including budgeting, accounting, financial reporting, and expense management.",
        "key_contacts": {
            "director": "Finance Director",
            "general_inquiries": "finance@company.com"
        },
        "common_topics": [
            "Expense reimbursement",
            "Budget planning",
            "Financial reports",
            "Tax information",
            "Payroll questions",
            "Invoice processing"
        ]
    },
    "Marketing": {
        "description": "The Marketing department handles brand management, marketing campaigns, market research, and communications strategy.",
        "key_contacts": {
            "director": "Marketing Director",
            "general_inquiries": "marketing@company.com"
        },
        "common_topics": [
            "Brand guidelines",
            "Marketing calendar",
            "Campaign performance",
            "Social media policies",
            "Press releases",
            "Event planning"
        ]
    },
    "IT": {
        "description": "The IT department provides technical support, manages infrastructure, and implements technology solutions across the organization.",
        "key_contacts": {
            "director": "IT Director",
            "general_inquiries": "it-support@company.com"
        },
        "common_topics": [
            "Technical support requests",
            "Software access and licensing",
            "Security policies",
            "Equipment requests",
            "Network issues",
            "System outages"
        ]
    },
    "Sales": {
        "description": "The Sales department manages client relationships, pursues new business opportunities, and meets revenue targets for the company.",
        "key_contacts": {
            "director": "Sales Director",
            "general_inquiries": "sales@company.com"
        },
        "common_topics": [
            "Sales targets and performance",
            "Client accounts",
            "CRM usage",
            "Sales presentations",
            "Pricing information",
            "Contract templates"
        ]
    }
}

# Department FAQs
DEPARTMENT_FAQS = {
    "HR": [
        {
            "question": "What is the company's work from home policy?",
            "answer": "The company allows employees to work from home up to 2 days per week, with manager approval. Arrangements must be documented in our HR system."
        },
        {
            "question": "How do I request time off?",
            "answer": "Time off requests should be submitted through the HR portal with at least 2 weeks advance notice for planned absences."
        },
        {
            "question": "When are performance reviews conducted?",
            "answer": "Performance reviews are conducted bi-annually, in June and December, with self-assessments due two weeks prior."
        }
    ],
    "Finance": [
        {
            "question": "What is the process for submitting expenses?",
            "answer": "Expenses should be submitted through the finance portal within 30 days of incurring the expense, with all receipts attached."
        },
        {
            "question": "When are expense reimbursements processed?",
            "answer": "Expense reimbursements are processed every Friday, with payments appearing in employee accounts within 3-5 business days."
        },
        {
            "question": "What's the fiscal year for the company?",
            "answer": "Our fiscal year runs from January 1 to December 31."
        }
    ],
    "Marketing": [
        {
            "question": "How do I access the brand guidelines?",
            "answer": "Brand guidelines are available on the Marketing team's SharePoint site, under Resources > Brand Assets."
        },
        {
            "question": "What is the process for requesting marketing materials?",
            "answer": "Requests for marketing materials should be submitted through the Marketing Request Form with at least 2 weeks lead time."
        },
        {
            "question": "Who approves social media posts?",
            "answer": "Social media posts require approval from the Social Media Manager and, for product-related content, the relevant Product Manager."
        }
    ],
    "IT": [
        {
            "question": "How do I reset my password?",
            "answer": "Passwords can be reset through the self-service portal at reset.company.com or by contacting the IT helpdesk."
        },
        {
            "question": "What is the process for requesting new software?",
            "answer": "New software requests should be submitted through the IT Service Portal with manager approval and business justification."
        },
        {
            "question": "How do I report a security incident?",
            "answer": "Security incidents should be reported immediately to security@company.com and your manager. For urgent issues, call the IT emergency line."
        }
    ],
    "Sales": [
        {
            "question": "Where can I find the latest sales presentations?",
            "answer": "The latest sales presentations are available in the Sales Resources folder on the shared drive, organized by product line and date."
        },
        {
            "question": "What is the discount approval process?",
            "answer": "Discounts up to 10% can be approved by Sales Managers. Discounts between 10-20% require Director approval. Anything above 20% needs VP approval."
        },
        {
            "question": "How do I access client history in the CRM?",
            "answer": "Client history can be accessed from the Account page in the CRM. Look for the 'Activity History' tab which shows all past interactions."
        }
    ]
}


# Function to get department information
def get_department_info(department):
    """
    Retrieve information about a specific department.
    
    Args:
        department (str): Department name
        
    Returns:
        dict: Department information
    """
    return DEPARTMENT_INFO.get(department, {})


# Function to get department FAQs
def get_department_faqs(department):
    """
    Retrieve FAQs for a specific department.
    
    Args:
        department (str): Department name
        
    Returns:
        list: Department FAQs
    """
    return DEPARTMENT_FAQS.get(department, [])


def authenticate_user(username, password):
    """
    Simulate authentication. In a real application, this would check against a database.
    
    Args:
        username (str): Username
        password (str): Password
        
    Returns:
        tuple: (is_authenticated (bool), user_role (str))
    """
    # This is a placeholder for a real authentication system
    # In production, use a secure authentication service
    
    # Mock authentication for demo purposes
    if username and password:
        # In a real application, determine the user's role from authentication response
        return True, "Employee"
    
    return False, None


def check_permission(action, user_role, department):
    """
    Check if user has permission to perform an action.
    
    Args:
        action (str): The action to check
        user_role (str): The user's role
        department (str): The department
        
    Returns:
        bool: True if user has permission, False otherwise
    """
    # Role-based access control
    # This is a simple implementation and should be expanded based on requirements
    
    if action == "view_basic_info":
        # All roles can view basic info
        return True
    
    elif action == "view_sensitive_info":
        # Only managers and above can view sensitive info
        return user_role in ["Manager", "Department Head", "Executive"]
    
    elif action == "modify_info":
        # Only department heads and executives can modify info
        return user_role in ["Department Head", "Executive"]
    
    elif action == "admin_actions":
        # Only executives can perform admin actions
        return user_role == "Executive"
    
    return False


class VectorStore:
    def __init__(self, db_path=VECTOR_DB_PATH):
        self.db_path = db_path
        self.dimension = 1536  # OpenAI embedding dimension
        self.index = None
        self.documents = {}
        self.document_ids = []
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize or load existing index
        self.init_index()
    
    def init_index(self):
        """Initialize FAISS index or load existing one."""
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        docs_path = os.path.join(self.db_path, "documents.pkl")
        
        try:
            if os.path.exists(index_path) and os.path.exists(docs_path):
                # Load existing index and documents
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.document_ids = data['document_ids']
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.dimension)
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            # Create new index as fallback
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def get_embedding(self, text):
        """Get OpenAI embedding for text."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            return None
    
    def add_document(self, text, metadata=None):
        """Add a document to the vector store."""
        try:
            # Generate document ID
            doc_id = hashlib.md5(text.encode()).hexdigest()
            
            # Get embedding
            embedding = self.get_embedding(text)
            if embedding is None:
                return None
            
            # Add to FAISS index
            embedding_np = np.array([embedding], dtype='float32')
            self.index.add(embedding_np)
            
            # Store document and metadata
            document_data = {
                'text': text,
                'metadata': metadata or {}
            }
            self.documents[doc_id] = document_data
            self.document_ids.append(doc_id)
            
            # Save updated index and documents
            self.save()
            
            return doc_id
        except Exception as e:
            st.error(f"Error adding document to vector store: {str(e)}")
            return None
    
    def search(self, query, k=5):
        """Search for similar documents to the query."""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Get query embedding
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                return []
            
            query_np = np.array([query_embedding], dtype='float32')
            
            # Search for similar embeddings
            distances, indices = self.index.search(query_np, min(k, self.index.ntotal))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.document_ids):  # Ensure index is valid
                    doc_id = self.document_ids[idx]
                    if doc_id in self.documents:
                        doc_data = self.documents[doc_id]
                        results.append({
                            'document': doc_data['text'],
                            'metadata': doc_data['metadata'],
                            'score': float(distances[0][i])
                        })
            
            return results
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
    
    def save(self):
        """Save the index and documents to disk."""
        try:
            index_path = os.path.join(self.db_path, "faiss_index.bin")
            docs_path = os.path.join(self.db_path, "documents.pkl")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save documents
            data = {
                'documents': self.documents,
                'document_ids': self.document_ids
            }
            with open(docs_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
    
    def get_document_by_id(self, doc_id):
        """Retrieve a document by its ID."""
        return self.documents.get(doc_id, None)


class WebhookHandler:
    def __init__(self):
        self.crm_webhook_url = os.environ.get("CRM_WEBHOOK_URL", "")
        self.mcp_webhook_url = os.environ.get("MCP_WEBHOOK_URL", "")
        self.api_key = os.environ.get("WEBHOOK_API_KEY", "")
    
    def fetch_crm_data(self, query_params):
        """
        Fetch data from the CRM system via webhook.
        
        Args:
            query_params (dict): Parameters for the CRM query
            
        Returns:
            dict: Response data or error information
        """
        if not self.crm_webhook_url:
            return {"error": "CRM webhook URL not configured"}
        
        try:
            # Add timestamp for request tracking
            query_params["timestamp"] = datetime.now().isoformat()
            
            # Make the request to the CRM webhook
            response = requests.post(
                self.crm_webhook_url,
                json=query_params,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Request-Source": "Knowledge-Hub"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"CRM webhook request failed with status {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            return {"error": f"Error connecting to CRM webhook: {str(e)}"}
    
    def fetch_mcp_data(self, query_params):
        """
        Fetch data from Microsoft Cloud Platform via webhook.
        
        Args:
            query_params (dict): Parameters for the MCP query
            
        Returns:
            dict: Response data or error information
        """
        if not self.mcp_webhook_url:
            return {"error": "MCP webhook URL not configured"}
        
        try:
            # Add timestamp for request tracking
            query_params["timestamp"] = datetime.now().isoformat()
            
            # Make the request to the MCP webhook
            response = requests.post(
                self.mcp_webhook_url,
                json=query_params,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Request-Source": "Knowledge-Hub"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"MCP webhook request failed with status {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            return {"error": f"Error connecting to MCP webhook: {str(e)}"}
    
    def fetch_employee_data(self, employee_id=None, email=None, department=None):
        """
        Fetch employee data from the CRM system.
        
        Args:
            employee_id (str, optional): Employee ID
            email (str, optional): Employee email
            department (str, optional): Department name
            
        Returns:
            dict: Employee data or error information
        """
        query_params = {
            "data_type": "employee",
            "employee_id": employee_id,
            "email": email,
            "department": department
        }
        
        return self.fetch_crm_data(query_params)
    
    def fetch_department_data(self, department_name):
        """
        Fetch department-specific data.
        
        Args:
            department_name (str): Name of the department
            
        Returns:
            dict: Department data or error information
        """
        query_params = {
            "data_type": "department",
            "department_name": department_name
        }
        
        return self.fetch_crm_data(query_params)
    
    def fetch_document_from_mcp(self, document_id=None, document_name=None):
        """
        Fetch document from Microsoft Cloud Platform.
        
        Args:
            document_id (str, optional): Document ID
            document_name (str, optional): Document name
            
        Returns:
            dict: Document data or error information
        """
        query_params = {
            "data_type": "document",
            "document_id": document_id,
            "document_name": document_name
        }
        
        return self.fetch_mcp_data(query_params)


def generate_response(prompt, context=None, department=None):
    """
    Generate a response using OpenAI's GPT-4o.
    
    Args:
        prompt (str): The user's query
        context (str, optional): Additional context from vector store
        department (str, optional): The department for context
        
    Returns:
        str: Generated response
    """
    messages = [
        {
            "role": "system", 
            "content": f"You are an Enterprise Knowledge Hub assistant specialized in providing information about {department if department else 'the company'}. "
                       f"Provide helpful, accurate, and concise answers based on company knowledge. "
                       f"If you don't know something, say so honestly instead of making up information."
        }
    ]
    
    # Add context from retrieval if available
    if context:
        messages.append({
            "role": "system", 
            "content": f"Here's some relevant information that may help answer the question: {context}"
        })
    
    # Add user prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def analyze_query(query):
    """
    Analyze the user query to determine intent and relevant department.
    
    Args:
        query (str): The user's query
        
    Returns:
        dict: Analysis result with intent and department
    """
    try:
        prompt = (
            "Analyze this query and determine: \n"
            "1. The primary intent (question_type: general_info, specific_document, data_request, process_help, technical_issue)\n"
            "2. The most relevant department (department: HR, Finance, Marketing, IT, Sales, General)\n"
            "Respond in JSON format."
        )
        
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        # Default to general information if analysis fails
        return {
            "question_type": "general_info",
            "department": "General"
        }


def analyze_document(text, document_name):
    """
    Analyze a document to extract key information and metadata.
    
    Args:
        text (str): Document text
        document_name (str): Name of the document
        
    Returns:
        dict: Document metadata and summary
    """
    try:
        prompt = (
            f"Analyze this document titled '{document_name}' and provide: \n"
            "1. A brief summary (1-3 sentences)\n"
            "2. Key topics covered (list of topics)\n"
            "3. Most relevant department (HR, Finance, Marketing, IT, Sales, General)\n"
            "4. Document type (policy, procedure, report, guide, other)\n"
            "Respond in JSON format."
        )
        
        # Truncate text if too long
        max_length = 8000
        if len(text) > max_length:
            text = text[:max_length] + "... (truncated)"
        
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        result["document_name"] = document_name
        return result
    except Exception as e:
        return {
            "summary": f"Error analyzing document: {str(e)}",
            "topics": [],
            "department": "General",
            "document_type": "unknown",
            "document_name": document_name
        }


def extract_text_from_pdf(file_content):
    """Extract text from PDF file."""
    pdf_file = io.BytesIO(file_content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_content):
    """Extract text from DOCX file."""
    docx_file = io.BytesIO(file_content)
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_csv(file_content):
    """Extract text from CSV file."""
    csv_file = io.BytesIO(file_content)
    df = pd.read_csv(csv_file)
    return df.to_string()


def process_document(uploaded_file, vector_store=None):
    """
    Process an uploaded document and extract its text.
    
    Args:
        uploaded_file (UploadedFile): The uploaded file from Streamlit
        vector_store (VectorStore, optional): Vector store to add the document to
        
    Returns:
        tuple: (document_text, metadata)
    """
    if vector_store is None:
        vector_store = VectorStore()
    
    try:
        # Get file content
        file_content = uploaded_file.read()
        file_extension = uploaded_file.name.split('.')[-1].lower()
        text = ""
        
        # Process based on file type
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension in ['doc', 'docx']:
            text = extract_text_from_docx(file_content)
        elif file_extension == 'txt':
            text = file_content.decode('utf-8')
        elif file_extension == 'csv':
            text = extract_text_from_csv(file_content)
        else:
            return None, {"error": f"Unsupported file format: {file_extension}"}
        
        if not text:
            return None, {"error": "Could not extract text from the document"}
        
        # Analyze document to get metadata
        metadata = analyze_document(text, uploaded_file.name)
        
        # Add document to vector store
        doc_id = vector_store.add_document(text, metadata)
        
        return text, metadata
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, {"error": str(e)}


def chunk_document(text, chunk_size=1000, overlap=100):
    """
    Split document text into overlapping chunks.
    
    Args:
        text (str): Document text
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end - start == chunk_size:
            # Find the last period or newline to make a clean cut
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            cut_point = max(last_period, last_newline)
            
            if cut_point > start + chunk_size // 2:  # Only use if it's reasonably far in
                end = cut_point + 1  # Include the period or newline
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def show_document_upload():
    """Show document upload interface in sidebar."""
    uploaded_file = st.file_uploader(
        "Upload a document to the knowledge base",
        type=["pdf", "docx", "txt", "csv"]
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            document_text, metadata = process_document(
                uploaded_file, 
                vector_store=st.session_state.vector_store
            )
            
            if document_text and not metadata.get("error"):
                st.success(f"Document '{uploaded_file.name}' added to knowledge base!")
                
                with st.expander("Document Analysis", expanded=False):
                    st.write("**Summary:**", metadata.get("summary", "Not available"))
                    st.write("**Department:**", metadata.get("department", "General"))
                    st.write("**Document Type:**", metadata.get("document_type", "Unknown"))
                    
                    st.write("**Key Topics:**")
                    for topic in metadata.get("topics", []):
                        st.write(f"- {topic}")
            else:
                st.error(f"Failed to process document: {metadata.get('error', 'Unknown error')}")


def create_sidebar():
    """Create and configure the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <style>
            .big-font {
                font-size:24px !important;
                font-weight: bold;
                color: #0078D4;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<p class="big-font">Enterprise Knowledge Hub</p>', unsafe_allow_html=True)
        
        # User information
        st.subheader("User Information")
        
        # Role selection (for demo purposes)
        selected_role = st.selectbox(
            "Select your role",
            options=USER_ROLES,
            index=USER_ROLES.index(st.session_state.user_role),
            key="role_selector"
        )
        
        # Update session state if role changed
        if selected_role != st.session_state.user_role:
            st.session_state.user_role = selected_role
        
        # Department selection
        st.subheader("Department")
        selected_department = st.selectbox(
            "Select department",
            options=DEPARTMENTS,
            index=DEPARTMENTS.index(st.session_state.selected_department),
            key="department_selector"
        )
        
        # Update session state if department changed
        if selected_department != st.session_state.selected_department:
            st.session_state.selected_department = selected_department
        
        # Display department information
        if selected_department:
            department_info = get_department_info(selected_department)
            if department_info:
                with st.expander(f"About {selected_department}", expanded=False):
                    st.write(department_info.get("description", "No description available."))
                    
                    st.subheader("Key Contacts")
                    for role, contact in department_info.get("key_contacts", {}).items():
                        st.write(f"**{role.title()}:** {contact}")
                    
                    st.subheader("Common Topics")
                    for topic in department_info.get("common_topics", []):
                        st.write(f"- {topic}")
        
        # Document upload section
        st.subheader("Document Management")
        
        # Only show upload if user has permission
        if check_permission("modify_info", st.session_state.user_role, selected_department):
            show_document_upload()
        else:
            st.info("You need higher permissions to upload documents.")
        
        # Clear chat history button
        st.subheader("Chat Options")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()


def display_chat_messages():
    """Display all chat messages."""
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        # Check if there are messages to display
        if not st.session_state.messages:
            return
        
        # Display all messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])


def process_user_input(user_input):
    """
    Process user input and generate a response.
    
    Args:
        user_input (str): The user's query
    """
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    st.chat_message("user").write(user_input)
    
    # Display thinking message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")
        
        # 1. Analyze the query using OpenAI
        query_analysis = analyze_query(user_input)
        
        # 2. Search for relevant information in vector store
        search_results = st.session_state.vector_store.search(user_input, k=3)
        context = ""
        if search_results:
            context = "\n\n".join([result["document"] for result in search_results])
        
        # 3. Check if we need to fetch data from external systems
        department = query_analysis.get("department", st.session_state.selected_department)
        webhook_data = None
        
        if query_analysis.get("question_type") == "data_request":
            # Determine which webhook to use based on the query
            if "CRM" in user_input or "customer" in user_input.lower() or "client" in user_input.lower():
                webhook_data = st.session_state.webhook_handler.fetch_crm_data({
                    "query": user_input,
                    "department": department
                })
            elif "MCP" in user_input or "Microsoft" in user_input or "document" in user_input.lower():
                webhook_data = st.session_state.webhook_handler.fetch_document_from_mcp(
                    document_name=user_input
                )
        
        # 4. Get department FAQs for additional context
        faqs = get_department_faqs(st.session_state.selected_department)
        faq_context = ""
        if faqs:
            faq_context = "Frequently Asked Questions:\n" + "\n".join([
                f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs
            ])
        
        # 5. Combine all context
        full_context = ""
        if context:
            full_context += f"Knowledge base information:\n{context}\n\n"
        if faq_context:
            full_context += f"{faq_context}\n\n"
        if webhook_data and not webhook_data.get("error"):
            full_context += f"External system data:\n{str(webhook_data)}\n\n"
        
        # 6. Generate response
        response = generate_response(
            user_input, 
            context=full_context, 
            department=st.session_state.selected_department
        )
        
        # 7. Update UI with response
        response_placeholder.markdown(response)
    
    # 8. Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 9. Add to chat history (for tracking complete conversations)
    st.session_state.chat_history.append({
        "query": user_input,
        "response": response,
        "department": st.session_state.selected_department,
        "timestamp": time.time()
    })


def create_chat_interface():
    """Create the chat interface component."""
    # Initialize the vector store if not already done
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    # Initialize webhook handler if not already done
    if "webhook_handler" not in st.session_state:
        st.session_state.webhook_handler = WebhookHandler()
    
    # Display chat messages
    display_chat_messages()
    
    # Chat input
    with st.container():
        st.write("")
        user_input = st.chat_input("Type your question here...")
        
        if user_input:
            process_user_input(user_input)


def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "selected_department" not in st.session_state:
        st.session_state.selected_department = DEPARTMENTS[0]
    
    if "user_role" not in st.session_state:
        st.session_state.user_role = USER_ROLES[2]  # Set default to "Department Head" for document upload access
    
    if "documents" not in st.session_state:
        st.session_state.documents = {}
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None


# Set page configuration
st.set_page_config(
    page_title="Enterprise Knowledge Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state variables
    initialize_session_state()
    
    # Create sidebar with department selection and user role
    create_sidebar()
    
    # Main page content
    st.title("Enterprise Knowledge Hub")
    
    # Display welcome message if no conversation started
    if not st.session_state.chat_history:
        st.markdown("""
        ### Welcome to the Enterprise Knowledge Hub
        
        I can help you with information about:
        - HR policies and procedures
        - Finance and accounting inquiries
        - Marketing resources and guidelines
        - IT support and technical documentation
        - Sales data and CRM information
        
        Just type your question to get started!
        """)
    
    # Create chat interface
    create_chat_interface()

if __name__ == "__main__":
    main()
