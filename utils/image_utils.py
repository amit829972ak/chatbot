# utils/image_utils.py
from PIL import Image
import io

def process_image(image):
    """
    Process and optimize an image for analysis.
    
    Args:
        image (PIL.Image): The image to process.
        
    Returns:
        PIL.Image: The processed image.
    """# utils/image_utils.py
from PIL import Image
import io

def process_image(image):
    """
    Process and optimize an image for analysis.
    
    Args:
        image (PIL.Image): The image to process.
        
    Returns:
        PIL.Image: The processed image.
    """
    # Ensure image is in RGB mode for compatibility
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if the image is too large (helps with API limits)
    max_size = 1600
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    
    # Return the processed image
    return image

def convert_image_to_bytes(image):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image (PIL.Image): The image to convert.
        
    Returns:
        bytes: The image as bytes.
    """
    if image is None:
        return None
        
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

def bytes_to_image(image_bytes):
    """
    Convert bytes to a PIL Image.
    
    Args:
        image_bytes (bytes): The image bytes.
        
    Returns:
        PIL.Image: The reconstructed image.
    """
    if image_bytes is None:
        return None
        
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None
    # Ensure image is in RGB mode for compatibility
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if the image is too large (helps with API limits)
    max_size = 1600
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    
    # Return the processed image
    return image

def convert_image_to_bytes(image):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image (PIL.Image): The image to convert.
        
    Returns:
        bytes: The image as bytes.
    """
    if image is None:
        return None
        
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

def bytes_to_image(image_bytes):
    """
    Convert bytes to a PIL Image.
    
    Args:
        image_bytes (bytes): The image bytes.
        
    Returns:
        PIL.Image: The reconstructed image.
    """
    if image_bytes is None:
        return None
        
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None
