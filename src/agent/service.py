import json
import logging
from uuid import UUID
import redis
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage

from . import models
from . import utils
from . import graph
from . import grammar
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
        # Initialize components
        components = utils.initialize_agent_components()
        
        # Create the graph
        agent_graph = graph.create_agent_graph(components, redis_client, db)
        
        # Create the config
        config = {
            "configurable": {
                "user_id": str(user_id),
                "conversation_id": str(conversation_id)
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
        except:
            # If conversation doesn't exist or there's an error, start with just the new message
            messages = []
        
        # Do language detection early to log
        lang = grammar.detect_language(message)
        logging.info(f"Message language detected: {lang}")
        
        # Add the current message
        messages.append(HumanMessage(content=message))
        
        # Initial state
        state = models.EnhancedState(
            messages=messages,
            question=message,
            answer="",
            memory={},
            context=[],
            search_needed=False,
            grammar_issues=None,
            corrected_text=None
        )
        
        # Run the graph
        result = agent_graph.invoke(state, config)
        
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
        memory_key = utils.get_memory_key(user_id)
        memory_json = redis_client.get(memory_key)
        updated_memory = json.loads(memory_json) if memory_json else None
        
        # Create the response object
        response = models.AgentResponse(
            message=last_message.content,
            updated_memory=models.UserProfileMemory(**updated_memory) if updated_memory else None
        )
        
        # Add grammar correction info if available
        if "grammar_correction" in result:
            response.grammar_correction = result["grammar_correction"]
        
        return response
    except Exception as e:
        logging.error(f"Error processing message through agent: {str(e)}")
        raise

def get_user_memory(redis_client: redis.Redis, user_id: UUID) -> models.AgentMemoryResponse:
    """
    Get the agent's memory for a user
    """
    try:
        # Retrieve memory from Redis
        memory_key = utils.get_memory_key(user_id)
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
        memory_key = utils.get_memory_key(user_id)
        redis_client.delete(memory_key)
        logging.info(f"Memory reset for user {user_id}")
    except Exception as e:
        logging.error(f"Error resetting memory for user {user_id}: {str(e)}")
        raise