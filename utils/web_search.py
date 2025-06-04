import streamlit as st
from duckduckgo_search import DDGS
from typing import List, Dict, Any

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
                    'title': result.get('title', ''),
                    'body': result.get('body', ''),
                    'href': result.get('href', ''),
                    'snippet': result.get('body', '')[:300] + '...' if len(result.get('body', '')) > 300 else result.get('body', '')
                })
            return results
    except Exception as e:
        st.error(f"Error during web search: {str(e)}")
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
    
    formatted_results = "**Web Search Results:**\n\n"
    
    for i, result in enumerate(results, 1):
        formatted_results += f"**{i}. {result['title']}**\n"
        formatted_results += f"{result['snippet']}\n"
        formatted_results += f"🔗 [Read more]({result['href']})\n\n"
    
    return formatted_results

def handle_web_search(query: str) -> str:
    """
    Handle web search request and return formatted results.
    
    Args:
        query: Search query string
        
    Returns:
        Formatted search results string
    """
    if not query.strip():
        return "Please provide a search query."
    
    # Perform web search
    results = search_web(query, max_results=5)
    
    if not results:
        return "No search results found for your query. Please try different keywords."
    
    # Format and return results
    return format_search_results(results)
