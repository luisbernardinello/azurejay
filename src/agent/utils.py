import json
import logging
from uuid import UUID
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults

# Constants
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses. Here is the memory (it may be empty): {memory}

If the question requires real-time information, I will search the internet and Wikipedia for you."""

TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

def get_memory_key(user_id: UUID) -> str:
    """Generate Redis key for storing agent memory for a user"""
    return f"agent:memory:{user_id}"

def initialize_llm(model_name: str = "gemini-1.5-flash", temperature: float = 0.2):
    """Initialize the language model"""
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def initialize_agent_components():
    """Initialize all the agent components"""
    # Initialize the LLM
    model = initialize_llm()
    
    # Create the components dict
    components = {
        "model": model
    }
    
    return components

def format_memory_for_prompt(memory: dict, first_name: str = None) -> str:
    """Format the user memory for inclusion in the system prompt"""
    if memory:
        return (
            f"Name: {memory.get('user_name', 'Unknown')}\n"
            f"Location: {memory.get('user_location', 'Unknown')}\n"
            f"Interests: {', '.join(memory.get('user_interests', []))}"
        )
    else:
        name_info = f"The user's name is: {first_name}." if first_name else ""
        return f"No memory available yet. Only the user's name is known. {name_info}"

def format_context_for_prompt(context: list) -> str:
    """Format search context for inclusion in the system prompt"""
    if not context:
        return ""
        
    combined_context = "\n\n".join(context)
    return f"\n\nHere's information I found that might help answer the question:\n{combined_context}"