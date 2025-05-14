import json
import logging
from uuid import UUID
import redis
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.store.redis import RedisStore, AsyncRedisStore
from langgraph.checkpoint.redis import RedisSaver, AsyncRedisSaver
from . import models
from . import utils
from . import graph
from ..conversations import service as conversation_service

def get_memory_key(user_id: UUID) -> str:
    """Generate a key for storing user memory in Redis"""
    return f"user_memory:{user_id}"

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
        # Initialize a Redis connection for LangGraph
        redis_uri = f"redis://{redis_client.connection_pool.connection_kwargs.get('host', 'localhost')}:{redis_client.connection_pool.connection_kwargs.get('port', 6379)}/0"
        
        # Setup LangGraph components
        with RedisSaver.from_conn_string(redis_uri) as checkpointer:
            checkpointer.setup()
            
            with RedisStore.from_conn_string(redis_uri) as store:
                store.setup()
                
                # Create the graph
                agent_graph = graph.create_agent_graph(store)
                
                # Create the config for the graph
                config = {
                    "configurable": {
                        "user_id": str(user_id),
                    }
                }
                
                # Create the input state with the user message
                input_messages = [HumanMessage(content=message)]
                
                # Run the graph
                result = agent_graph.invoke({"messages": input_messages}, config)
                
                # Extract the response (last message)
                last_message = result["messages"][-1]
                response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                # Create the response object
                response = models.AgentResponse(
                    message=response_content,
                    updated_memory=None  # We'll populate this in get_user_memory
                )
                
                # Check if there's grammar correction info
                namespace = ("grammar", str(user_id))
                grammar_memories = store.search(namespace)
                
                if grammar_memories:
                    # Sort by timestamp to get the most recent correction
                    corrections = [mem.value for mem in grammar_memories]
                    sorted_corrections = sorted(
                        corrections, 
                        key=lambda x: x.get('timestamp', ''), 
                        reverse=True
                    )
                    
                    if sorted_corrections:
                        latest = sorted_corrections[0]
                        response.grammar_correction = {
                            "original_text": latest.get("original_text", ""),
                            "corrected_text": latest.get("corrected_text", ""),
                            "explanation": latest.get("explanation", "")
                        }
                
                return response
                
    except Exception as e:
        logging.error(f"Error processing message through agent: {str(e)}", exc_info=True)
        raise

def get_user_memory(redis_client: redis.Redis, user_id: UUID) -> models.AgentMemoryResponse:
    """
    Get the agent's memory for a user
    """
    try:
        # Initialize a Redis connection for LangGraph
        redis_uri = f"redis://{redis_client.connection_pool.connection_kwargs.get('host', 'localhost')}:{redis_client.connection_pool.connection_kwargs.get('port', 6379)}/0"
        
        # Setup LangGraph components
        with RedisSaver.from_conn_string(redis_uri) as checkpointer:
            # Retrieve memory components from Redis store
            with RedisStore.from_conn_string(redis_uri) as store:
                
                profile_data = None
                topics_data = []
                grammar_data = []
                web_search_data = {}
                
                # Get profile memory
                profile_memories = store.search(("profile", str(user_id)))
                if profile_memories:
                    profile_data = profile_memories[0].value
                
                # Get topics memory
                topic_memories = store.search(("topic", str(user_id)))
                if topic_memories:
                    topics_data = [mem.value for mem in topic_memories]
                
                # Get grammar memory
                grammar_memories = store.search(("grammar", str(user_id)))
                if grammar_memories:
                    grammar_data = [mem.value for mem in grammar_memories]
                
                # Get web search memory
                web_search_memories = store.search(("web_search", str(user_id)))
                if web_search_memories:
                    web_search_data = web_search_memories[0].value
                
                # Create a UserProfileMemory object
                memory = models.UserProfileMemory(
                    profile=profile_data,
                    topics=topics_data,
                    grammar_corrections=grammar_data,
                    web_search=web_search_data
                )
                
                return models.AgentMemoryResponse(
                    user_id=user_id,
                    memory=memory
                )
    except Exception as e:
        logging.error(f"Error retrieving memory for user {user_id}: {str(e)}", exc_info=True)
        raise

def reset_user_memory(redis_client: redis.Redis, user_id: UUID) -> None:
    """
    Reset (delete) the agent's memory for a user
    """
    try:
        # Initialize a Redis connection for LangGraph
        redis_uri = f"redis://{redis_client.connection_pool.connection_kwargs.get('host', 'localhost')}:{redis_client.connection_pool.connection_kwargs.get('port', 6379)}/0"
        
        # Setup LangGraph components
        with RedisSaver.from_conn_string(redis_uri) as checkpointer:
            
            with RedisStore.from_conn_string(redis_uri) as store:

                # Delete profile memory
                profile_memories = store.search(("profile", str(user_id)))
                for mem in profile_memories:
                    store.delete(("profile", str(user_id)), mem.key)
                    
                # Delete topic memory
                topic_memories = store.search(("topic", str(user_id)))
                for mem in topic_memories:
                    store.delete(("topic", str(user_id)), mem.key)
                    
                # Delete grammar memory
                grammar_memories = store.search(("grammar", str(user_id)))
                for mem in grammar_memories:
                    store.delete(("grammar", str(user_id)), mem.key)
                    
                # Delete web search memory
                web_search_memories = store.search(("web_search", str(user_id)))
                for mem in web_search_memories:
                    store.delete(("web_search", str(user_id)), mem.key)
                
        logging.info(f"Memory reset for user {user_id}")
    except Exception as e:
        logging.error(f"Error resetting memory for user {user_id}: {str(e)}", exc_info=True)
        raise