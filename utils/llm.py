import streamlit as st
import os

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
    # First check if API key is provided
    if not api_key:
        return "Please provide a valid API key in the sidebar to use this model."
    
    try:
        # Get the specific model name from session state
        specific_model = st.session_state.get("specific_model", "")
        
        if model_choice == "OpenAI GPT":
            return query_openai(prompt, api_key, specific_model)
        elif model_choice == "Google Gemini":
            return query_gemini(prompt, api_key, specific_model)
        elif model_choice == "Claude":
            return query_claude(prompt, api_key, specific_model)
        else:
            return "Unknown model selected."
    except Exception as e:
        return f"Error querying {model_choice}: {str(e)}"

def query_openai(prompt, api_key, specific_model="OpenAI GPT-4o"):
    """Query OpenAI's GPT model."""
    try:
        from openai import OpenAI
        
        # Map display names to actual model identifiers
        model_mapping = {
            "OpenAI GPT-4o": "gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            "OpenAI GPT-4": "gpt-4",
            "OpenAI GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        
        # Get the actual model identifier to use
        model_id = model_mapping.get(specific_model, "gpt-4o")
        
        # Initialize the client
        client = OpenAI(api_key=api_key)
        
        # Create the completion
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful company assistant that provides factual information based on company documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=1000
        )
        
        # Return the response text
        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "No response generated from the model."
    
    except Exception as e:
        return f"Error with OpenAI API: {str(e)}"

def query_gemini(prompt, api_key, specific_model="Google Gemini Pro"):
    """Query Google's Gemini model."""
    try:
        import google.generativeai as genai
        import os
        
        # Map display names to actual model identifiers
        model_mapping = {
            "Google Gemini Pro": "gemini-pro",
            "Google Gemini Flash": "gemini-flash",
            "Google Gemini 1.0 Pro Vision": "gemini-1.0-pro-vision",
            "Google Gemini 1.5 Pro": "gemini-1.5-pro",
            "Google Gemini 1.5 Flash": "gemini-1.5-flash",
            "Google Gemini 1.5 Pro Latest": "gemini-1.5-pro-latest",
            "Google Gemini 1.5 Flash Latest": "gemini-1.5-flash-latest",
            "Google Gemini 2.0 Pro Vision": "gemini-2.0-pro-vision",
            "Google Gemini 2.0 Pro": "gemini-2.0-pro",
            "Google Gemini 2.5 Pro": "gemini-2.5-pro",
            "Google Gemini 2.5 Flash": "gemini-2.5-flash"
        }
        
        # Get the actual model identifier to use
        model_id = model_mapping.get(specific_model, "gemini-pro")
        
        # Try different API configuration methods to handle different versions
        try:
            # Method 1: Newer versions use configure
            if hasattr(genai, 'configure'):
                genai.configure(api_key=api_key)
            # Method 2: Some versions use _configure
            elif hasattr(genai, '_configure'):
                genai._configure(api_key=api_key)
            # Method 3: Fallback to environment variable
            else:
                os.environ["GOOGLE_API_KEY"] = api_key
        except Exception as config_error:
            # Final fallback for environment variable
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Create a model instance - handle different parameter naming
        try:
            # Try the first method with model_name parameter
            model = genai.GenerativeModel(model_name=model_id)
        except:
            try:
                # Try alternative method with model parameter
                model = genai.GenerativeModel(model=model_id)
            except:
                # Final fallback using positional argument
                model = genai.GenerativeModel(model_id)
        
        # Generate content with safety settings for better compatability
        try:
            response = model.generate_content(prompt)
        except TypeError:
            # Some versions need additional parameters or different methods
            try:
                response = model.generate_content(contents=prompt)
            except:
                # Last resort fallback
                response = model.generate_content(
                    prompt, 
                    generation_config={"temperature": 0.3, "max_output_tokens": 1000}
                )
        
        # Extract response text with multiple fallbacks for robustness
        response_text = ""
        
        # Try all known response formats systematically
        if hasattr(response, 'text'):
            response_text = str(response.text)
        elif hasattr(response, 'parts'):
            parts = []
            for part in response.parts:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(part['text'])
                elif hasattr(part, 'text') and part.text:
                    parts.append(str(part.text))
                elif isinstance(part, str):
                    parts.append(part)
            response_text = ''.join(parts)
        elif hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    response_text = str(candidate.content)
                    break
        elif hasattr(response, 'result'):
            response_text = str(response.result)
        elif hasattr(response, 'generations') and response.generations:
            response_text = str(response.generations[0].text)
            
        # Last resort is to convert the entire response to string
        if not response_text:
            response_text = str(response)
            
        return response_text
    
    except Exception as e:
        return f"Error with Google Gemini API: {str(e)}"

def query_claude(prompt, api_key, specific_model="Claude 3.5 Sonnet"):
    """Query Anthropic's Claude model."""
    try:
        import anthropic
        import json
        
        # Map display names to actual model identifiers
        model_mapping = {
            "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",  # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            "Claude 3 Opus": "claude-3-opus-20240229",
            "Claude 3 Sonnet": "claude-3-sonnet-20240229",
            "Claude 3 Haiku": "claude-3-haiku-20240307"
        }
        
        # Get the actual model identifier to use
        model_id = model_mapping.get(specific_model, "claude-3-5-sonnet-20241022")
        
        # Initialize the client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Create the message with safe parameter handling
        try:
            message = client.messages.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}  # Use dict instead of MessageParam for better compatibility
                ]
            )
        except TypeError:
            # Fallback for different API versions
            message = client.messages.create(
                model=model_id,
                max_tokens=1024,
                system="You are a helpful company assistant that provides factual information based on company documents.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        
        # Extract response text using multiple methods for robustness
        response_text = ""
        
        try:
            # For API response with content list
            if hasattr(message, 'content'):
                # Convert content to list if it's not already
                content_list = message.content if isinstance(message.content, list) else [message.content]
                
                for item in content_list:
                    # Try different ways to extract text
                    if isinstance(item, dict) and 'text' in item:
                        response_text += item['text']
                    elif hasattr(item, 'text'):
                        response_text += str(item.text)
                    elif isinstance(item, str):
                        response_text += item
            
            # For older API versions with completion attribute
            if not response_text and hasattr(message, 'completion'):
                response_text = str(message.completion)
            
            # For newer API with content-type fields
            if not response_text and hasattr(message, 'type'):
                if message.type == 'text':
                    response_text = str(message.text)
                elif message.type == 'message' and hasattr(message, 'value'):
                    response_text = str(message.value)
            
            # Last resort: try to extract from the response dictionary
            if not response_text:
                if hasattr(message, '__dict__'):
                    message_dict = message.__dict__
                    if 'text' in message_dict:
                        response_text = str(message_dict['text'])
                    elif 'content' in message_dict:
                        content = message_dict['content']
                        if isinstance(content, list) and len(content) > 0:
                            first_item = content[0]
                            if isinstance(first_item, dict) and 'text' in first_item:
                                response_text = str(first_item['text'])
            
            # If all extraction methods fail, convert the whole response to string
            if not response_text:
                try:
                    response_text = str(message)
                except:
                    response_text = "Could not extract text from Claude response."
            
            return response_text
            
        except Exception as extract_error:
            return f"Error extracting text from Claude response: {str(extract_error)}"
    
    except Exception as e:
        return f"Error with Claude API: {str(e)}"
