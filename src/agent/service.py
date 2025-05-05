import json
import logging
import os
from uuid import UUID
from dotenv import load_dotenv
import redis

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import MemoryStore
from trustcall import create_extractor

from . import models
from ..conversations import service as conversation_service

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
# Constants
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses. Here is the memory (it may be empty): {memory}"""

TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

# Memory key patterns
def get_memory_key(user_id: UUID) -> str:
    """Generate Redis key for storing agent memory for a user"""
    return f"agent:memory:{user_id}"


def initialize_agent(model_name: str = "gemini-1.5-flash", temperature: float = 0.2):
    """Initialize the agent components"""
    # Initialize the LLM
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
    # Create the memory extractor
    trustcall_extractor = create_extractor(
        model,
        tools=[models.UserProfileMemory],
        tool_choice="UserProfileMemory"  # Enforces use of the UserProfileMemory tool
    )
    
    # Store these in a dict to pass around
    components = {
        "model": model,
        "extractor": trustcall_extractor
    }
    
    return components


def call_model(state: MessagesState, config: RunnableConfig, components: dict, redis_client: redis.Redis):
    """
    Load memory from Redis and use it to personalize the chatbot's response.
    """
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    conversation_id = config["configurable"]["conversation_id"]
    
    # Retrieve memory from Redis
    memory_key = get_memory_key(user_id)
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
        formatted_memory = "No memory available yet."
    
    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)
    
    # Respond using memory as well as the chat history
    response = components["model"].invoke([SystemMessage(content=system_msg)] + state["messages"])
    
    # Log the response for debugging
    logging.info(f"Generated response for user {user_id}, conversation {conversation_id}")
    
    return {"messages": state["messages"] + [response]}


def write_memory(state: MessagesState, config: RunnableConfig, components: dict, redis_client: redis.Redis):
    """
    Reflect on the chat history and save memory to Redis.
    """
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    
    # Retrieve existing memory from Redis
    memory_key = get_memory_key(user_id)
    memory_json = redis_client.get(memory_key)
    existing_memory = json.loads(memory_json) if memory_json else None
    
    # Prepare the existing profile for the extractor
    existing_profile = {"UserProfileMemory": existing_memory} if existing_memory else None
    
    # Invoke the extractor
    result = components["extractor"].invoke({
        "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"], 
        "existing": existing_profile
    })
    
    # Get the updated profile as a JSON object
    updated_profile = result["responses"][0].model_dump()
    
    # Save the updated profile to Redis
    redis_client.set(memory_key, json.dumps(updated_profile))
    logging.info(f"Updated memory for user {user_id}")
    
    return {"messages": state["messages"], "memory": updated_profile}


def create_agent_graph(components: dict, redis_client: redis.Redis):
    """
    Create the langgraph for the agent
    """
    # Define the nodes that will make up our graph
    def call_model_with_components(state, config):
        return call_model(state, config, components, redis_client)
    
    def write_memory_with_components(state, config):
        return write_memory(state, config, components, redis_client)
    
    # Create the graph
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model_with_components)
    builder.add_node("write_memory", write_memory_with_components)
    
    # Define the edges
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "write_memory")
    builder.add_edge("write_memory", END)
    
    # Compile the graph
    return builder.compile()


def process_message(
    redis_client: redis.Redis, 
    user_id: UUID, 
    conversation_id: UUID, 
    message: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0
) -> models.AgentResponse:
    """
    Process a user message through the agent graph
    """
    try:
        # Initialize or get components
        components = initialize_agent(model_name, temperature)
        
        # Create the graph
        graph = create_agent_graph(components, redis_client)
        
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
        state = {"messages": messages}
        
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