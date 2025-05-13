import json
import logging
from uuid import UUID
from typing import Literal, Dict, Any
import redis
from sqlalchemy.orm import Session

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from . import models
from . import configuration
from . import utils
from . import search
from . import grammar
from ..users.service import get_user_by_name
from trustcall import create_extractor

def route_message(state: models.EnhancedState) -> Literal["search_web", "generate_response", "grammar_correction"]:
    """
    Conditional routing function to determine whether to search, correct grammar, or directly answer.
    """
    # Extract the last user message
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return "generate_response"  # If not a human message, just respond
        
    query = last_message.content.lower()
    
    # First check if grammar correction is needed
    if grammar.needs_grammar_correction(last_message.content):

        if grammar.check_grammar(last_message.content)[0]:  # If there are grammar issues
            logging.info(f"Grammar issues detected. Routing to grammar_correction.")
            return "grammar_correction"
    
    # Then check if search is needed
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
        return "generate_response"

def generate_response(state: models.EnhancedState, config: RunnableConfig, components: dict, redis_client: redis.Redis, db: Session):
    """
    Generate a response using memory and context.
    (Renamed from call_model to better reflect its purpose)
    """
    # Get the user ID from the config
    config_obj = configuration.Configuration.from_runnable_config(config)
    user_id = config_obj.user_id
    conversation_id = config_obj.conversation_id
    
    # Retrieve memory from Redis
    memory_key = utils.get_memory_key(UUID(user_id))
    memory_json = redis_client.get(memory_key)
    memory = json.loads(memory_json) if memory_json else None
    
    # Get user's first name if no memory exists
    first_name = None
    if not memory:
        first_name = get_user_by_name(db, UUID(user_id))
    
    # Format the memories for the system prompt
    formatted_memory = utils.format_memory_for_prompt(memory, first_name)
    
    # Format the memory in the system prompt
    system_msg = utils.MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)
    
    # Add language instruction to ensure English responses
    system_msg += "\nIMPORTANT: You should ONLY respond in English, even if the user writes in another language."
    
    # Add context from search results if available
    context = state.get("context", [])
    if context:
        context_instruction = utils.format_context_for_prompt(context)
        system_msg += context_instruction
    
    # Respond using memory as well as the chat history
    response = components["model"].invoke([SystemMessage(content=system_msg)] + state["messages"])
    
    # Log the response for debugging
    logging.info(f"Generated response for user {user_id}, conversation {conversation_id}")
    
    return {"messages": state["messages"] + [response], "answer": response.content}

def write_memory(state: models.EnhancedState, config: RunnableConfig, redis_client: redis.Redis):
    """
    Reflect on the chat history and save memory to Redis.
    """
    # Get the user ID from the config
    config_obj = configuration.Configuration.from_runnable_config(config)
    user_id = config_obj.user_id
    
    # Initialize the LLM for memory extraction
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    trustcall_extractor = create_extractor(
        model,
        tools=[models.UserProfileMemory],
        tool_choice="UserProfileMemory"  # Enforces use of the UserProfileMemory tool
    )
    
    # Retrieve existing memory from Redis
    memory_key = utils.get_memory_key(UUID(user_id))
    memory_json = redis_client.get(memory_key)
    existing_memory = json.loads(memory_json) if memory_json else None
    
    # Prepare the existing profile for the extractor
    existing_profile = {"UserProfileMemory": existing_memory} if existing_memory else None
    
    # Invoke the extractor
    result = trustcall_extractor.invoke({
        "messages": [SystemMessage(content=utils.TRUSTCALL_INSTRUCTION)] + state["messages"], 
        "existing": existing_profile
    })
    
    # Get the updated profile as a JSON object
    updated_profile = result["responses"][0].model_dump()
    
    # Save the updated profile to Redis
    redis_client.set(memory_key, json.dumps(updated_profile))
    logging.info(f"Updated memory for user {user_id}")
    
    return {"messages": state["messages"], "memory": updated_profile}

def create_agent_graph(components: dict, redis_client: redis.Redis, db: Session):
    """
    Create the langgraph for the agent with a more explicit routing mechanism
    """
    # Define the nodes that will make up our graph
    def generate_response_with_deps(state, config):
        return generate_response(state, config, components, redis_client, db)
    
    def write_memory_with_deps(state, config):
        return write_memory(state, config, redis_client)
    
    def search_web_node(state, config):
        return search.search_web(state, config)
    
    def search_wikipedia_node(state, config):
        return search.search_wikipedia(state, config)
    
    def grammar_correction_node(state, config):
        # First process grammar to detect issues
        grammar_state = grammar.process_grammar(state)
        
        # Update the state with grammar information
        updated_state = {**state, **grammar_state}
        
        # Generate a response about grammar issues
        return grammar.generate_grammar_response(updated_state, config, components)
    
    # Create the graph
    builder = StateGraph(models.EnhancedState)
    
    # Add nodes
    builder.add_node("generate_response", generate_response_with_deps)
    builder.add_node("write_memory", write_memory_with_deps)
    builder.add_node("search_web", search_web_node)
    builder.add_node("search_wikipedia", search_wikipedia_node)
    builder.add_node("grammar_correction", grammar_correction_node)
    
    # Define the starting edge using route_message to determine flow
    builder.add_conditional_edges(
        START,
        route_message  # Use the route_message function to decide the first node
    )
    
    # Grammar correction leads to memory writing
    builder.add_edge("grammar_correction", "write_memory")
    
    # Search web leads to Wikipedia search
    builder.add_edge("search_web", "search_wikipedia")
    
    # Wikipedia search leads to model call
    builder.add_edge("search_wikipedia", "generate_response")
    
    # Model call leads to memory writing
    builder.add_edge("generate_response", "write_memory")
    
    # Memory writing ends the graph
    builder.add_edge("write_memory", END)
    
    # Compile the graph
    return builder.compile()