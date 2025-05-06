from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

def create_rag_chain(api_key=None):
    """
    Create a RAG chain using LangChain with Google Gemini.
    
    Args:
        api_key (str, optional): Google Gemini API key.
        
    Returns:
        LLMChain: A LangChain chain for RAG responses.
    """
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    # Create a RAG prompt template
    rag_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant. Use the following context to answer the user's question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question using the context provided. If the context doesn't contain the information needed,
        acknowledge this and provide a response based on your general knowledge.
        """
    )
    
    # Create the chain
    chain = LLMChain(
        llm=llm,
        prompt=rag_prompt
    )
    
    return chain

def get_rag_response(query, context, api_key=None):
    """
    Get a RAG-enhanced response using the provided context.
    
    Args:
        query (str): The user's query.
        context (list): List of relevant information from the vector store.
        api_key (str, optional): Google Gemini API key.
        
    Returns:
        str: The RAG-enhanced response.
    """
    # Create the chain
    chain = create_rag_chain(api_key)
    
    # Format the context
    formatted_context = ""
    if context:
        for item in context:
            formatted_context += f"- {item['content']}\n"
    else:
        formatted_context = "No relevant context found."
    
    # Get the response
    response = chain.run(context=formatted_context, question=query)
    
    return response

def get_multimodal_response(query, image_analysis, api_key=None):
    """
    Get a response that incorporates both text query and image analysis.
    
    Args:
        query (str): The user's text query.
        image_analysis (str): Analysis of the uploaded image.
        api_key (str, optional): Google Gemini API key.
        
    Returns:
        str: Response that considers both the text and image.
    """
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    # Create a multimodal prompt template
    multimodal_prompt = PromptTemplate(
        input_variables=["image_analysis", "query"],
        template="""
        You are a helpful assistant. A user has uploaded an image and asked a question about it.
        
        Image Analysis:
        {image_analysis}
        
        User Query:
        {query}
        
        Provide a helpful response that addresses the user's query based on the image analysis.
        """
    )
    
    # Create the chain
    chain = LLMChain(
        llm=llm,
        prompt=multimodal_prompt
    )
    
    # Get the response
    response = chain.run(image_analysis=image_analysis, query=query)
    
    return response
