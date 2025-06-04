import streamlit as st
import hashlib
import os
import json
from datetime import datetime, timedelta

# File to store admin credentials
CREDENTIALS_FILE = "admin_credentials.json"

def initialize_credentials():
    """
    Initialize default admin credentials if credentials file doesn't exist.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        # Default admin credentials
        default_password = "admin123"  # This is just a default, should be changed after first login
        hashed_password = hashlib.sha256(default_password.encode()).hexdigest()
        
        credentials = {
            "admin": {
                "password": hashed_password,
                "created_at": datetime.now().isoformat()
            }
        }
        
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(credentials, f, indent=4)

def verify_credentials(username, password):
    """
    Verify admin credentials.
    
    Args:
        username: Username to verify
        password: Password to verify
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Initialize credentials if not already done
    if not os.path.exists(CREDENTIALS_FILE):
        initialize_credentials()
    
    # Read credentials from file
    with open(CREDENTIALS_FILE, "r") as f:
        credentials = json.load(f)
    
    # Check if username exists
    if username not in credentials:
        return False
    
    # Hash the provided password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    # Compare with stored password
    return hashed_password == credentials[username]["password"]

def change_password(username, new_password):
    """
    Change admin password.
    
    Args:
        username: Username to change password for
        new_password: New password to set
        
    Returns:
        bool: True if password was changed successfully, False otherwise
    """
    # Initialize credentials if not already done
    if not os.path.exists(CREDENTIALS_FILE):
        initialize_credentials()
    
    # Read credentials from file
    with open(CREDENTIALS_FILE, "r") as f:
        credentials = json.load(f)
    
    # Check if username exists
    if username not in credentials:
        return False
    
    # Hash the new password
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    
    # Update the password
    credentials[username]["password"] = hashed_password
    credentials[username]["updated_at"] = datetime.now().isoformat()
    
    # Write credentials back to file
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f, indent=4)
    
    return True

def login_form():
    """
    Display login form in Streamlit sidebar.
    
    Updates session state 'authenticated' to True if login is successful.
    """
    # Initialize credentials if not already done
    if not os.path.exists(CREDENTIALS_FILE):
        initialize_credentials()
    
    st.sidebar.subheader("Admin Login")
    
    # Login form
    username = st.sidebar.text_input("Username", value="admin")
    password = st.sidebar.text_input("Password", type="password")
    
    # Login button
    if st.sidebar.button("Login"):
        if verify_credentials(username, password):
            st.session_state['authenticated'] = True
            st.sidebar.success("Login successful!")
        else:
            st.sidebar.error("Invalid username or password")

def change_password_form(username):
    """
    Display change password form in Streamlit sidebar.
    
    Args:
        username: Username to change password for
    """
    st.sidebar.subheader("Change Password")
    
    # Change password form
    new_password = st.sidebar.text_input("New Password", type="password", key="new_pass")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="confirm_pass")
    
    # Change password button
    if st.sidebar.button("Change Password"):
        if new_password != confirm_password:
            st.sidebar.error("Passwords do not match")
        elif len(new_password) < 6:
            st.sidebar.error("Password must be at least 6 characters")
        else:
            if change_password(username, new_password):
                st.sidebar.success("Password changed successfully!")
            else:
                st.sidebar.error("Failed to change password")
