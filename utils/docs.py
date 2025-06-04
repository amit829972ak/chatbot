import streamlit as st
import os
import tempfile
from utils.vector_store import update_index_from_file, remove_from_index

UPLOAD_DIR = "uploaded_docs"

def handle_upload():
    """Handle document upload through sidebar."""
    
    # Initialize session state for upload process
    if 'upload_state' not in st.session_state:
        st.session_state['upload_state'] = {
            'message': "",
            'processing': False
        }
    
    # Custom CSS for clean upload button
    st.sidebar.markdown("""
    <style>
    .upload-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 10px 0;
    }
    .upload-button {
        width: 60px;
        height: 60px;
        border: 2px dashed #0EA5E9;
        border-radius: 8px;
        background-color: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }
    .upload-button:hover {
        background-color: #f0f9ff;
        border-color: #0b8bcf;
    }
    .upload-plus {
        font-size: 24px;
        color: #0EA5E9;
        font-weight: bold;
    }
    .upload-text {
        font-size: 12px;
        color: #64748b;
        text-align: center;
        margin-top: 5px;
    }
    div[data-testid="stFileUploader"] > div > div > div > button {
        display: none;
    }
    div[data-testid="stFileUploader"] > div > div > small {
        display: none;
    }
    div[data-testid="stFileUploader"] > div > div {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # File uploader widget with new formats
    file = st.sidebar.file_uploader(
        "+", 
        type=["pdf", "txt", "docx", "xlsx", "xls", "pptx", "ppt"], 
        accept_multiple_files=False,
        help="Upload documents to be indexed: PDF, TXT, DOCX, XLSX, PPTX",
        key="doc_uploader",
        label_visibility="collapsed"
    )
    
    # Display upload button only when a file is selected
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if file is not None:
            upload_clicked = st.button("📤 Upload", help="Upload and process the document")
        else:
            upload_clicked = False
    
    # Handle file upload
    if upload_clicked and file is not None:
        st.session_state['upload_state']['processing'] = True
        
        try:
            # Create upload directory if it doesn't exist
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Process the file and update the vector index
            with st.spinner(f"Processing {file.name}..."):
                success = update_index_from_file(file_path)
            
            if success:
                st.session_state['upload_state'] = {
                    'message': f"✅ Successfully uploaded and indexed: {file.name}",
                    'processing': False
                }
                st.success(f"Successfully uploaded and indexed: {file.name}")
                st.rerun()
            else:
                st.session_state['upload_state'] = {
                    'message': f"❌ Failed to process: {file.name}",
                    'processing': False
                }
                st.error(f"Failed to process: {file.name}")
                
                # Remove the file if processing failed
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
        except Exception as e:
            st.session_state['upload_state'] = {
                'message': f"❌ Error uploading {file.name}: {str(e)}",
                'processing': False
            }
            st.error(f"Error uploading {file.name}: {str(e)}")
    
    # Display upload status
    if st.session_state['upload_state']['message']:
        if "✅" in st.session_state['upload_state']['message']:
            st.sidebar.success(st.session_state['upload_state']['message'])
        else:
            st.sidebar.error(st.session_state['upload_state']['message'])

def handle_delete():
    """Handle document deletion through sidebar."""
    st.sidebar.subheader("Delete Documents")
    
    documents = list_documents()
    if documents:
        # Initialize session state for deletion
        if 'delete_state' not in st.session_state:
            st.session_state['delete_state'] = {
                'selected_doc': None,
                'confirm_delete': False,
                'message': ""
            }
        
        # Document selection
        selected_doc = st.sidebar.selectbox(
            "Select document to delete:",
            options=["Select a document..."] + documents,
            key="delete_selectbox"
        )
        
        if selected_doc != "Select a document...":
            st.session_state['delete_state']['selected_doc'] = selected_doc
            
            # Confirmation checkbox
            confirm = st.sidebar.checkbox(
                f"Confirm deletion of '{selected_doc}'",
                key="delete_confirmation"
            )
            
            if confirm:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("🗑️ Delete", key="delete_button"):
                        try:
                            file_path = os.path.join(UPLOAD_DIR, selected_doc)
                            
                            # Remove from vector index
                            with st.spinner(f"Removing {selected_doc} from index..."):
                                index_success = remove_from_index(file_path)
                            
                            # Remove physical file
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                file_success = True
                            else:
                                file_success = False
                            
                            if index_success and file_success:
                                st.session_state['delete_state'] = {
                                    'selected_doc': None,
                                    'confirm_delete': False,
                                    'message': f"✅ Successfully deleted: {selected_doc}"
                                }
                                st.success(f"Successfully deleted: {selected_doc}")
                                st.rerun()
                            else:
                                st.session_state['delete_state']['message'] = f"❌ Failed to delete: {selected_doc}"
                                st.error(f"Failed to delete: {selected_doc}")
                                
                        except Exception as e:
                            st.session_state['delete_state']['message'] = f"❌ Error deleting {selected_doc}: {str(e)}"
                            st.error(f"Error deleting {selected_doc}: {str(e)}")
                with col2:
                    if st.button("❌ Cancel", key="cancel_delete"):
                        st.session_state['delete_state'] = {
                            'selected_doc': None,
                            'confirm_delete': False,
                            'message': ""
                        }
    else:
        st.sidebar.info("No documents available to delete.")

def list_documents():
    """Return list of available documents."""
    if os.path.exists(UPLOAD_DIR):
        return sorted(os.listdir(UPLOAD_DIR))
    return []

def get_document_versions():
    """Get document version information."""
    try:
        from utils.vector_store import AdvancedVectorStore
        vector_store = AdvancedVectorStore()
        return vector_store.versions
    except Exception as e:
        print(f"Error getting document versions: {e}")
        return {}

def display_document_info():
    """Display document information with versioning."""
    versions_info = get_document_versions()
    
    if versions_info:
        st.sidebar.subheader("Document Versions")
        
        for doc_name, version_data in versions_info.items():
            current_version = version_data.get('current_version', 1)
            total_versions = len(version_data.get('versions', []))
            
            if total_versions > 1:
                st.sidebar.info(f"📄 {doc_name}\nVersion: {current_version}/{total_versions}")
                
                # Show version details in expander
                with st.sidebar.expander(f"Version History - {doc_name}"):
                    for version in reversed(version_data.get('versions', [])):
                        timestamp = version.get('timestamp', 'Unknown')
                        size = version.get('size', 0)
                        v_num = version.get('version', 1)
                        
                        st.write(f"**Version {v_num}**")
                        st.write(f"Date: {timestamp[:19].replace('T', ' ')}")
                        st.write(f"Size: {size:,} characters")
                        st.write("---")
            else:
                st.sidebar.info(f"📄 {doc_name}\nVersion: {current_version}")
    else:
        st.sidebar.info("No document versions available")
