import os
import io
import base64
from PIL import Image
import json
from openai import OpenAI

def get_openai_client(api_key=None):
    """
    Get or create an OpenAI client with the provided API key.
    If no key is provided, try to use the environment variable.
    
    Args:
        api_key (str, optional): OpenAI API key
        
    Returns:
        OpenAI: OpenAI client
    """
    # Use provided key, fallback to environment variable
    key = api_key or os.environ.get("OPENAI_API_KEY")
    
    # Create the client
    return OpenAI(api_key=key)

def get_ai_response(prompt, system_prompt=None, context=None, api_key=None):
    """
    Get a response from the OpenAI API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): OpenAI API key.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error (rate limit, authentication, etc.)
    """
    try:
        client = get_openai_client(api_key)
        
        # Prepare messages
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation context if available
        if context:
            for message in context:
                messages.append({"role": message["role"], "content": message["content"]})
        
        # Add the user's prompt
        messages.append({"role": "user", "content": prompt})
        
        # Get the response
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        if "Rate limit" in error_msg:
            raise Exception("OpenAI API rate limit reached. Please try again later.")
        elif "Authentication" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        else:
            raise Exception(f"Error generating response: {error_msg}")

def encode_image_to_base64(image):
    """
    Encode an image to base64 for API transmission.
    
    Args:
        image (PIL.Image): The image to encode.
        
    Returns:
        str: Base64-encoded image string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_image_content(image, api_key=None):
    """
    Analyze an image using OpenAI's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): OpenAI API key.
        
    Returns:
        str: Analysis of the image content.
    """
    try:
        client = get_openai_client(api_key)
        
        # Convert image to base64
        base64_image = encode_image_to_base64(image)
        
        # Create the request
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4o as it supports multimodal input
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image in detail. Describe what you see and provide any relevant information about the content."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        if "Rate limit" in error_msg:
            raise Exception("OpenAI API rate limit reached. Please try again later.")
        elif "Authentication" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        else:
            raise Exception(f"Error analyzing image: {error_msg}")

def get_embedding(text, api_key=None):
    """
    Get an embedding vector for the given text.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): OpenAI API key.
        
    Returns:
        list: The embedding vector.
    """
    try:
        client = get_openai_client(api_key)
        
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        
        return response.data[0].embedding
    
    except Exception as e:
        error_msg = str(e)
        if "Rate limit" in error_msg:
            raise Exception("OpenAI API rate limit reached. Please try again later.")
        elif "Authentication" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        else:
            raise Exception(f"Error generating embedding: {error_msg}")
