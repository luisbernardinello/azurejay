import json
import logging
from uuid import UUID
import redis
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage

from . import models
from . import utils
from . import graph
from ..conversations import service as conversation_service

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
        # Setup and retrieve the agent graph
        agent_graph = graph.setup_and_run_graph()
        
        # Create the config
        config = {
            "configurable": {
                "user_id": str(user_id),
                "thread_id": str(conversation_id)
            }
        }
        
        # Get existing conversation messages from Redis
        try:
            conv_detail = conversation_service.get_conversation(redis_client, user_id, conversation_id)
            messages = []
            for msg in conv_detail.messages:
                if msg.is_user:
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))
        except Exception as e:
            logging.info(f"Starting new conversation: {str(e)}")
            # If conversation doesn't exist or there's an error, start with just the new message
            messages = []
        
        # Add the current message
        messages.append(HumanMessage(content=message))
        
        # Run the graph with the current conversation state
        response_chunks = []
        for chunk in agent_graph.stream({"messages": messages}, config, stream_mode="values"):
            response_chunks.append(chunk)
        
        # Get the last message (assistant's response)
        if response_chunks and "messages" in response_chunks[-1]:
            last_message = response_chunks[-1]["messages"][-1]
            assistant_response = last_message.content
        else:
            logging.error("No response received from agent graph")
            assistant_response = "I apologize, but I couldn't process your message right now."
        
        # Store the user message in the conversation
        message_request = conversation_service.models.MessageRequest(content=message)
        conversation_service.add_message(redis_client, user_id, conversation_id, message_request, is_user=True)
        
        # Store the AI response in the conversation
        ai_message_request = conversation_service.models.MessageRequest(content=assistant_response)
        conversation_service.add_message(redis_client, user_id, conversation_id, ai_message_request, is_user=False)
        
        # Get the user memory
        user_memory = get_user_memory(redis_client, user_id)
        
        # Create the response object
        response = models.AgentResponse(
            message=assistant_response,
            updated_memory=user_memory.memory
        )
        
        return response
    except Exception as e:
        logging.error(f"Error processing message through agent: {str(e)}")
        raise

def get_user_memory(redis_client: redis.Redis, user_id: UUID) -> models.AgentMemoryResponse:
    """
    Get the agent's memory for a user by aggregating all memory types
    """
    try:
        memory_data = {
            "profile": None,
            "topics": [],
            "grammar_corrections": [],
            "web_search": None
        }
        
        # Convert user_id to string for Redis store
        user_id_str = str(user_id)
        
        # Try to get profile memory
        profile_key = ("profile", user_id_str)
        profile_memories = utils.search_redis_memories(redis_client, profile_key)
        if profile_memories and len(profile_memories) > 0:
            memory_data["profile"] = profile_memories[0]
        
        # Get topic memories
        topic_key = ("topic", user_id_str)
        topic_memories = utils.search_redis_memories(redis_client, topic_key)
        if topic_memories:
            memory_data["topics"] = topic_memories
        
        # Get grammar correction memories
        grammar_key = ("grammar", user_id_str)
        grammar_memories = utils.search_redis_memories(redis_client, grammar_key)
        if grammar_memories:
            memory_data["grammar_corrections"] = grammar_memories
        
        # Get web search memories
        web_search_key = ("web_search", user_id_str)
        web_search_memories = utils.search_redis_memories(redis_client, web_search_key)
        if web_search_memories and len(web_search_memories) > 0:
            memory_data["web_search"] = web_search_memories[0]
        
        # Create UserProfileMemory object
        user_profile_memory = models.UserProfileMemory(
            profile=memory_data["profile"],
            topics=memory_data["topics"],
            grammar_corrections=memory_data["grammar_corrections"],
            web_search=memory_data["web_search"]
        )
        
        return models.AgentMemoryResponse(
            user_id=user_id,
            memory=user_profile_memory
        )
    except Exception as e:
        logging.error(f"Error retrieving memory for user {user_id}: {str(e)}")
        raise

def reset_user_memory(redis_client: redis.Redis, user_id: UUID) -> None:
    """
    Reset (delete) all agent memory types for a user
    """
    try:
        user_id_str = str(user_id)
        
        # Delete all memory types
        memory_types = ["profile", "topic", "grammar", "web_search"]
        for memory_type in memory_types:
            key = (memory_type, user_id_str)
            utils.delete_redis_memories(redis_client, key)
        
        logging.info(f"All memory types reset for user {user_id}")
    except Exception as e:
        logging.error(f"Error resetting memory for user {user_id}: {str(e)}")
        raise