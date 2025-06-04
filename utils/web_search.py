from typing import List, Dict, Any
from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, body, and href
    """
    try:
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', 'No title'),
                    'body': result.get('body', 'No description'),
                    'href': result.get('href', '#')
                })
            return results
    except Exception as e:
        print(f"Error performing web search: {e}")
        return []

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a readable string.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string with search results
    """
    if not results:
        return "No search results found."
    
    formatted = "**Web Search Results:**\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        body = result.get('body', 'No description')
        url = result.get('href', '#')
        
        formatted += f"**{i}. {title}**\n"
        formatted += f"{body}\n"
        formatted += f"Source: {url}\n\n"
    
    return formatted

def handle_web_search(query: str) -> str:
    """
    Handle web search request and return formatted results.
    
    Args:
        query: Search query string
        
    Returns:
        Formatted search results string
    """
    try:
        # Perform web search
        results = search_web(query, max_results=5)
        
        # Format and return results
        return format_search_results(results)
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"
