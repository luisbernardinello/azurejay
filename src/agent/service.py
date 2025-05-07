import json
import logging
import os
import operator
from uuid import UUID
from typing import Annotated, List, Dict, Any, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
import redis

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, MessagesState, START, END

from src.users.service import get_user_by_name
from sqlalchemy.orm import Session

from . import models
from . import configuration
from ..conversations import service as conversation_service

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Constants
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses. Here is the memory (it may be empty): {memory}

If the question requires real-time information, I will search the internet and Wikipedia for you."""

TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

# Memory key patterns
def get_memory_key(user_id: UUID) -> str:
    """Generate Redis key for storing agent memory for a user"""
    return f"agent:memory:{user_id}"


# Enhanced state that includes context for search results and routing information
class EnhancedState(TypedDict):
    messages: list
    question: str
    answer: str
    memory: Dict[str, Any]
    context: Annotated[list, operator.add]
    search_needed: bool  # Track if search is needed


# Tool choice definition for routing
class RouteDecision(TypedDict):
    """ Decision on what route to take in the agent graph """
    route: Literal['search', 'direct_answer', 'memory_update']


def initialize_agent():
    """Initialize the agent components"""
    # Initialize the LLM
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    # Create search tools
    tavily_search = TavilySearchResults(max_results=1)
    
    # Store these in a dict to pass around
    components = {
        "model": model,
        "tavily_search": tavily_search
    }
    
    return components


def search_web(state: EnhancedState, config: RunnableConfig, components: dict):
    """Retrieve docs from web search"""
    
    # Extract question from the last message
    last_message = state["messages"][-1]
    question = last_message.content
    
    # Search
    search_docs = components["tavily_search"].invoke(question)
    
    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    logging.info(f"Web search results: {formatted_search_docs}")
    return {"context": [formatted_search_docs]}


def search_wikipedia(state: EnhancedState, config: RunnableConfig):
    """Retrieve docs from wikipedia"""
    
    # Extract question from the last message
    last_message = state["messages"][-1]
    question = last_message.content
    
    try:
        # Search
        search_docs = WikipediaLoader(query=question, load_max_docs=1).load()
        
        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        logging.info(f"Wikipedia search results: {formatted_search_docs}")
        return {"context": [formatted_search_docs]}
    except Exception as e:
        logging.warning(f"Wikipedia search failed: {str(e)}")
        return {"context": ["No relevant Wikipedia results found."]}


def call_model(state: EnhancedState, config: RunnableConfig, components: dict, redis_client: redis.Redis, db: Session):
    """
    Load memory from Redis and use it to personalize the chatbot's response.
    Also takes into account retrieved context from searches.
    """
    # Get the user ID from the config
    config_obj = configuration.Configuration.from_runnable_config(config)
    user_id = config_obj.user_id
    conversation_id = config_obj.conversation_id
    
    # Retrieve memory from Redis
    memory_key = get_memory_key(UUID(user_id))
    memory_json = redis_client.get(memory_key)
    memory = json.loads(memory_json) if memory_json else None
    
    # Format the memories for the system prompt
    if memory:
        formatted_memory = (
            f"Name: {memory.get('user_name', 'Unknown')}\n"
            f"Location: {memory.get('user_location', 'Unknown')}\n"
            f"Interests: {', '.join(memory.get('user_interests', []))}"
        )
    else: 
        first_name = get_user_by_name(db, UUID(user_id))
        formatted_memory = f"No memory available yet. Only the user's name is known. The user's name is: {first_name}."
    
    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)
    
    # Add context from search results if available
    context = state.get("context", [])
    if context:
        combined_context = "\n\n".join(context)
        context_instruction = f"\n\nHere's information I found that might help answer the question:\n{combined_context}"
        system_msg += context_instruction
    
    # Respond using memory as well as the chat history
    response = components["model"].invoke([SystemMessage(content=system_msg)] + state["messages"])
    
    # Log the response for debugging
    logging.info(f"Generated response for user {user_id}, conversation {conversation_id}")
    
    return {"messages": state["messages"] + [response], "answer": response.content}


def write_memory(state: EnhancedState, config: RunnableConfig, redis_client: redis.Redis):
    """
    Reflect on the chat history and save memory to Redis.
    """
    # Get the user ID from the config
    config_obj = configuration.Configuration.from_runnable_config(config)
    user_id = config_obj.user_id
    
    # Initialize the LLM for memory extraction
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    from trustcall import create_extractor
    trustcall_extractor = create_extractor(
        model,
        tools=[models.UserProfileMemory],
        tool_choice="UserProfileMemory"  # Enforces use of the UserProfileMemory tool
    )
    
    # Retrieve existing memory from Redis
    memory_key = get_memory_key(UUID(user_id))
    memory_json = redis_client.get(memory_key)
    existing_memory = json.loads(memory_json) if memory_json else None
    
    # Prepare the existing profile for the extractor
    existing_profile = {"UserProfileMemory": existing_memory} if existing_memory else None
    
    # Invoke the extractor
    result = trustcall_extractor.invoke({
        "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"], 
        "existing": existing_profile
    })
    
    # Get the updated profile as a JSON object
    updated_profile = result["responses"][0].model_dump()
    
    # Save the updated profile to Redis
    redis_client.set(memory_key, json.dumps(updated_profile))
    logging.info(f"Updated memory for user {user_id}")
    
    return {"messages": state["messages"], "memory": updated_profile}


def route_message(state: EnhancedState) -> Literal["search_web", "call_model"]:
    """
    Conditional routing function to determine whether to search or directly answer.

    """
    # Extract the last user message
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return "call_model"  # If not a human message, just respond
        
    query = last_message.content.lower()
    
    # Check if query likely needs external information
    search_indicators = [
        "what is", "who is", "when did", "where is", "how does", 
        "latest", "recent", "news", "information about",
        "tell me about", "search for", "find information", 
        "can you find", "look up", "research"
    ]
    
    # If any search indicator is present, use search
    if any(indicator in query for indicator in search_indicators):
        return "search_web"
    else:
        return "call_model"


def create_agent_graph(components: dict, redis_client: redis.Redis, db: Session):
    """
    Create the langgraph for the agent with a explicit routing mechanism

    """
    # Define the nodes that will make up our graph
    def call_model_with_components(state, config):
        return call_model(state, config, components, redis_client, db)
    
    def write_memory_with_components(state, config):
        return write_memory(state, config, redis_client)
    
    def search_web_with_components(state, config):
        return search_web(state, config, components)
    
    def search_wikipedia_with_components(state, config):
        return search_wikipedia(state, config)
    
    # Create the graph
    builder = StateGraph(EnhancedState)
    
    # Add nodes
    builder.add_node("call_model", call_model_with_components)
    builder.add_node("write_memory", write_memory_with_components)
    builder.add_node("search_web", search_web_with_components)
    builder.add_node("search_wikipedia", search_wikipedia_with_components)
    
    # Define the starting edge using route_message to determine flow
    builder.add_conditional_edges(
        START,
        route_message  # Use the route_message function to decide the first node
    )
    
    # Search web leads to Wikipedia search
    builder.add_edge("search_web", "search_wikipedia")
    
    # Wikipedia search leads to model call
    builder.add_edge("search_wikipedia", "call_model")
    
    # Model call leads to memory writing
    builder.add_edge("call_model", "write_memory")
    
    # Memory writing ends the graph
    builder.add_edge("write_memory", END)
    
    # Compile the graph
    return builder.compile()


def process_message(
    redis_client: redis.Redis,
    db: Session, 
    user_id: UUID, 
    conversation_id: UUID, 
    message: str,
) -> models.AgentResponse:
    """
    Process a user message through the agent graph
    """
    try:
        # Initialize or get components
        components = initialize_agent()
        
        # Create the graph
        graph = create_agent_graph(components, redis_client, db)
        
        # Create the config
        config = {
            "configurable": {
                "user_id": str(user_id),
                "conversation_id": str(conversation_id)
            }
        }
        
        # Create the message state
        # Get existing conversation messages from Redis
        try:
            conv_detail = conversation_service.get_conversation(redis_client, user_id, conversation_id)
            messages = []
            for msg in conv_detail.messages:
                if msg.is_user:
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))
        except:
            # If conversation doesn't exist or there's an error, start with just the new message
            messages = []
        
        # Add the current message
        messages.append(HumanMessage(content=message))
        
        # Initial state
        state = EnhancedState(
            messages=messages,
            question=message,
            answer="",
            memory={},
            context=[],
            search_needed=False  # Initialize the search flag
        )
        
        # Run the graph
        result = graph.invoke(state, config)
        
        # Extract the response
        response_messages = result["messages"]
        last_message = response_messages[-1]
        
        # Store the response message in the conversation
        message_request = conversation_service.models.MessageRequest(content=message)
        conversation_service.add_message(redis_client, user_id, conversation_id, message_request, is_user=True)
        
        # Store the AI response in the conversation
        ai_message_request = conversation_service.models.MessageRequest(content=last_message.content)
        conversation_service.add_message(redis_client, user_id, conversation_id, ai_message_request, is_user=False)
        
        # Get the updated memory
        memory_key = get_memory_key(user_id)
        memory_json = redis_client.get(memory_key)
        updated_memory = json.loads(memory_json) if memory_json else None
        
        # Return the response
        return models.AgentResponse(
            message=last_message.content,
            updated_memory=models.UserProfileMemory(**updated_memory) if updated_memory else None
        )
    except Exception as e:
        logging.error(f"Error processing message through agent: {str(e)}")
        raise


def get_user_memory(redis_client: redis.Redis, user_id: UUID) -> models.AgentMemoryResponse:
    """
    Get the agent's memory for a user
    """
    try:
        # Retrieve memory from Redis
        memory_key = get_memory_key(user_id)
        memory_json = redis_client.get(memory_key)
        memory = json.loads(memory_json) if memory_json else None
        
        return models.AgentMemoryResponse(
            user_id=user_id,
            memory=models.UserProfileMemory(**memory) if memory else None
        )
    except Exception as e:
        logging.error(f"Error retrieving memory for user {user_id}: {str(e)}")
        raise


def reset_user_memory(redis_client: redis.Redis, user_id: UUID) -> None:
    """
    Reset (delete) the agent's memory for a user
    """
    try:
        # Delete memory from Redis
        memory_key = get_memory_key(user_id)
        redis_client.delete(memory_key)
        logging.info(f"Memory reset for user {user_id}")
    except Exception as e:
        logging.error(f"Error resetting memory for user {user_id}: {str(e)}")
        raise