import streamlit as st
import json
import os
import hashlib

CREDENTIALS_FILE = "admin_credentials.json"

def initialize_credentials():
    """
    Initialize default admin credentials if credentials file doesn't exist.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        default_credentials = {
            "admin": hashlib.sha256("admin123".encode()).hexdigest()
        }
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(default_credentials, f)

def verify_credentials(username, password):
    """
    Verify admin credentials.
    
    Args:
        username: Username to verify
        password: Password to verify
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        initialize_credentials()
        
        with open(CREDENTIALS_FILE, 'r') as f:
            credentials = json.load(f)
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        return username in credentials and credentials[username] == password_hash
    except Exception as e:
        st.error(f"Error verifying credentials: {e}")
        return False

def change_password(username, new_password):
    """
    Change admin password.
    
    Args:
        username: Username to change password for
        new_password: New password to set
        
    Returns:
        bool: True if password was changed successfully, False otherwise
    """
    try:
        initialize_credentials()
        
        with open(CREDENTIALS_FILE, 'r') as f:
            credentials = json.load(f)
        
        if username in credentials:
            credentials[username] = hashlib.sha256(new_password.encode()).hexdigest()
            
            with open(CREDENTIALS_FILE, 'w') as f:
                json.dump(credentials, f)
            
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error changing password: {e}")
        return False

def login_form():
    """
    Display login form in Streamlit sidebar.
    
    Updates session state 'authenticated' to True if login is successful.
    """
    st.sidebar.subheader("Admin Login")
    
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if verify_credentials(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

def change_password_form(username):
    """
    Display change password form in Streamlit sidebar.
    
    Args:
        username: Username to change password for
    """
    st.sidebar.subheader("Change Password")
    
    with st.sidebar.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Change Password")
        
        if submitted:
            if verify_credentials(username, current_password):
                if new_password == confirm_password:
                    if len(new_password) >= 6:
                        if change_password(username, new_password):
                            st.success("Password changed successfully!")
                        else:
                            st.error("Failed to change password")
                    else:
                        st.error("Password must be at least 6 characters long")
                else:
                    st.error("New passwords do not match")
            else:
                st.error("Current password is incorrect")
