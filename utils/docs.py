import streamlit as st
import os
from utils_vector_store import update_index_from_file, remove_from_index

# Directory to store uploaded documents
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def handle_upload():
    """Handle document upload through sidebar."""
    st.sidebar.subheader("Upload Document")
    
    # Initialize session state for upload process
    if 'upload_state' not in st.session_state:
        st.session_state['upload_state'] = {
            'file_uploaded': False,
            'message': "",
            'processing': False
        }
    
    # File uploader widget
    file = st.sidebar.file_uploader(
        "Select file", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=False,
        help="Upload PDF, TXT, or DOCX files to be indexed for RAG",
        key="doc_uploader"
    )
    
    # Display upload button only when a file is selected
    col1, col2 = st.sidebar.columns(2)
    with col1:
        upload_button = st.button(
            "Upload and Index",
            key="upload_doc_button",
            disabled=file is None or st.session_state['upload_state']['processing']
        )
    
    # Handle file processing on button click (without page reload)
    if upload_button and file is not None and not st.session_state['upload_state']['processing']:
        # Mark as processing to prevent multiple submissions
        st.session_state['upload_state']['processing'] = True
        
        try:
            # Show processing indicator
            status_placeholder = st.sidebar.empty()
            status_placeholder.info("Processing document... please wait")
            
            # Create file path and save file
            filepath = os.path.join(UPLOAD_DIR, file.name)
            with open(filepath, "wb") as f:
                f.write(file.getvalue())
            
            # Process and index the file
            update_index_from_file(filepath)
            
            # Update status
            st.session_state['upload_state']['file_uploaded'] = True
            st.session_state['upload_state']['message'] = f"📄 {file.name} uploaded and indexed!"
            status_placeholder.success(st.session_state['upload_state']['message'])
            
        except Exception as e:
            error_msg = f"Error uploading document: {str(e)}"
            st.session_state['upload_state']['message'] = error_msg
            st.sidebar.error(error_msg)
            
        finally:
            # Mark processing as complete
            st.session_state['upload_state']['processing'] = False
    
    # Show status messages based on upload state
    elif st.session_state['upload_state']['file_uploaded'] and not upload_button:
        if "Error" in st.session_state['upload_state']['message']:
            st.sidebar.error(st.session_state['upload_state']['message'])
        else:
            st.sidebar.success(st.session_state['upload_state']['message'])
    
    # Button to clear the upload status
    with col2:
        clear_button = st.button(
            "Clear Status",
            key="clear_upload_status",
            disabled=not st.session_state['upload_state']['file_uploaded']
        )
        
    if clear_button:
        # Reset the upload state
        st.session_state['upload_state'] = {
            'file_uploaded': False,
            'message': "",
            'processing': False
        }

def handle_delete():
    """Handle document deletion through sidebar."""
    st.sidebar.subheader("Delete Document")
    
    # Initialize session state for delete process
    if 'delete_state' not in st.session_state:
        st.session_state['delete_state'] = {
            'file_deleted': False,
            'message': "",
            'processing': False
        }
    
    # List available documents
    files = list_documents()
    if files:
        # Document selection dropdown
        selected = st.sidebar.selectbox(
            "Select document to delete", 
            files, 
            key="delete_select"
        )
        
        # Delete button and clear status in columns
        col1, col2 = st.sidebar.columns(2)
        with col1:
            delete_button = st.button(
                "Delete Document", 
                key="delete_doc_button",
                disabled=st.session_state['delete_state']['processing']
            )
        
        # Handle document deletion on button click
        if delete_button and not st.session_state['delete_state']['processing']:
            # Mark as processing to prevent multiple deletions
            st.session_state['delete_state']['processing'] = True
            
            try:
                # Show processing indicator
                status_placeholder = st.sidebar.empty()
                status_placeholder.info("Deleting document...")
                
                # Delete the file
                file_path = os.path.join(UPLOAD_DIR, selected)
                remove_from_index(file_path)
                os.remove(file_path)
                
                # Update status
                st.session_state['delete_state']['file_deleted'] = True
                st.session_state['delete_state']['message'] = f"🗑️ {selected} deleted"
                status_placeholder.success(st.session_state['delete_state']['message'])
                
            except Exception as e:
                error_msg = f"Error deleting document: {str(e)}"
                st.session_state['delete_state']['message'] = error_msg
                st.sidebar.error(error_msg)
                
            finally:
                # Mark processing as complete
                st.session_state['delete_state']['processing'] = False
        
        # Show status messages based on delete state
        elif st.session_state['delete_state']['file_deleted'] and not delete_button:
            if "Error" in st.session_state['delete_state']['message']:
                st.sidebar.error(st.session_state['delete_state']['message'])
            else:
                st.sidebar.success(st.session_state['delete_state']['message'])
        
        # Button to clear the delete status
        with col2:
            clear_button = st.button(
                "Clear Status",
                key="clear_delete_status",
                disabled=not st.session_state['delete_state']['file_deleted']
            )
            
        if clear_button:
            # Reset the delete state
            st.session_state['delete_state'] = {
                'file_deleted': False,
                'message': "",
                'processing': False
            }
    else:
        st.sidebar.info("No documents available to delete.")

def list_documents():
    """Return list of available documents."""
    if os.path.exists(UPLOAD_DIR):
        return sorted(os.listdir(UPLOAD_DIR))
    return []
