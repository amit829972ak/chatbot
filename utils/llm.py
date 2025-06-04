import os
import streamlit as st

def query_model(prompt, model_choice, api_key):
    """
    Query the selected AI model with the given prompt.
    
    Args:
        prompt: The prompt to send to the AI
        model_choice: Which AI model to use
        api_key: API key for the selected model
        
    Returns:
        Response text from the AI model
    """
    if not api_key:
        return "Please provide an API key in the sidebar to use AI features."
    
    try:
        specific_model = st.session_state.get("specific_model", model_choice)
        
        if model_choice == "OpenAI GPT":
            return query_openai(prompt, api_key, specific_model)
        elif model_choice == "Google Gemini":
            return query_gemini(prompt, api_key, specific_model)
        elif model_choice == "Claude":
            return query_claude(prompt, api_key, specific_model)
        else:
            return "Unsupported model type. Please select OpenAI GPT, Google Gemini, or Claude."
    
    except Exception as e:
        return f"Error querying AI model: {str(e)}"

def query_openai(prompt, api_key, specific_model="OpenAI GPT-4o"):
    """Query OpenAI's GPT model."""
    try:
        import openai
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Map display names to API model names
        model_mapping = {
            "OpenAI GPT-4o": "gpt-4o",
            "OpenAI GPT-4": "gpt-4",
            "OpenAI GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        
        model_name = model_mapping.get(specific_model, "gpt-4o")
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate and informative responses based on the given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "Invalid OpenAI API key. Please check your API key and try again."
        elif "quota" in error_msg.lower():
            return "OpenAI API quota exceeded. Please check your account limits."
        else:
            return f"OpenAI API error: {error_msg}"

def query_gemini(prompt, api_key, specific_model="Google Gemini Pro"):
    """Query Google's Gemini model."""
    try:
        import google.generativeai as genai
        
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Map display names to API model names
        model_mapping = {
            "Google Gemini Pro": "gemini-pro",
            "Google Gemini Flash": "gemini-1.5-flash",
            "Google Gemini 1.0 Pro Vision": "gemini-pro-vision",
            "Google Gemini 1.5 Pro": "gemini-1.5-pro",
            "Google Gemini 1.5 Flash": "gemini-1.5-flash",
            "Google Gemini 1.5 Pro Latest": "gemini-1.5-pro-latest",
            "Google Gemini 1.5 Flash Latest": "gemini-1.5-flash-latest",
            "Google Gemini 2.0 Pro Vision": "gemini-2.0-pro-vision",
            "Google Gemini 2.0 Pro": "gemini-2.0-pro",
            "Google Gemini 2.5 Pro": "gemini-2.5-pro",
            "Google Gemini 2.5 Flash": "gemini-2.5-flash"
        }
        
        model_name = model_mapping.get(specific_model, "gemini-pro")
        
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "Invalid Google API key. Please check your API key and try again."
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "Google API quota exceeded. Please check your account limits."
        else:
            return f"Google Gemini API error: {error_msg}"

def query_claude(prompt, api_key, specific_model="Claude 3.5 Sonnet"):
    """Query Anthropic's Claude model."""
    try:
        import anthropic
        
        # Initialize the client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Map display names to API model names
        model_mapping = {
            "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
            "Claude 3 Opus": "claude-3-opus-20240229",
            "Claude 3 Sonnet": "claude-3-sonnet-20240229",
            "Claude 3 Haiku": "claude-3-haiku-20240307"
        }
        
        model_name = model_mapping.get(specific_model, "claude-3-5-sonnet-20241022")
        
        # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        response = client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "Invalid Anthropic API key. Please check your API key and try again."
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "Anthropic API quota exceeded. Please check your account limits."
        else:
            return f"Anthropic Claude API error: {error_msg}"
