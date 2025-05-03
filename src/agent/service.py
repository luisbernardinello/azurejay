from datetime import datetime, timedelta
import json
import redis
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from .utils import initialize_model, extract_profile_from_conversation
from .models import ChatRequest, ChatResponse, Profile, Conversation
from .agent_graph import agent_graph

# Initialize model
model = initialize_model()

# Initialize Redis connection
# You should move this to environment variables
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
CONVERSATION_EXPIRY = 60 * 60 * 24 * 7  # 7 days in seconds

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logging.info("Successfully connected to Redis")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {str(e)}")
    redis_client = None

def store_conversation(user_id: UUID, user_message: str, agent_response: str) -> bool:
    """Store a conversation in Redis."""
    if not redis_client:
        logging.error("Redis client not available")
        return False
    
    try:
        # Create conversation record
        conversation = Conversation(
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=agent_response
        )
        
        # Store in Redis with expiration
        conversation_key = f"conversation:{user_id}:{datetime.now().isoformat()}"
        redis_client.set(
            conversation_key,
            json.dumps(conversation.model_dump()),
            ex=CONVERSATION_EXPIRY
        )
        return True
    except Exception as e:
        logging.error(f"Error storing conversation: {str(e)}")
        return False

def get_recent_conversations(user_id: UUID, limit: int = 5) -> List[Conversation]:
    """Get recent conversations for a user from Redis."""
    if not redis_client:
        logging.error("Redis client not available")
        return []
    
    try:
        # Get all keys for this user's conversations
        pattern = f"conversation:{user_id}:*"
        conversation_keys = redis_client.keys(pattern)
        
        # Sort by timestamp (which is in the key)
        conversation_keys.sort(reverse=True)
        
        # Get the most recent conversations
        recent_keys = conversation_keys[:limit]
        conversations = []
        
        for key in recent_keys:
            conversation_data = redis_client.get(key)
            if conversation_data:
                try:
                    conversation = Conversation(**json.loads(conversation_data))
                    conversations.append(conversation)
                except Exception as e:
                    logging.error(f"Error parsing conversation data: {str(e)}")
        
        return conversations
    except Exception as e:
        logging.error(f"Error retrieving conversations: {str(e)}")
        return []

def get_user_profile(user_id: UUID) -> Optional[Profile]:
    """Get the user profile from Redis."""
    if not redis_client:
        logging.error("Redis client not available")
        return None
    
    try:
        profile_key = f"profile:{user_id}"
        profile_data = redis_client.get(profile_key)
        
        if profile_data:
            return Profile(**json.loads(profile_data))
        return None
    except Exception as e:
        logging.error(f"Error retrieving user profile: {str(e)}")
        return None

def store_user_profile(user_id: UUID, profile: Profile) -> bool:
    """Store the user profile in Redis."""
    if not redis_client:
        logging.error("Redis client not available")
        return False
    
    try:
        profile_key = f"profile:{user_id}"
        redis_client.set(
            profile_key,
            json.dumps(profile.model_dump())
        )  # No expiration for profiles
        return True
    except Exception as e:
        logging.error(f"Error storing user profile: {str(e)}")
        return False

def update_profile_if_needed(user_id: UUID, conversation_history: List) -> bool:
    """Check if we need to update the user profile and do so if needed."""
    try:
        # Get current profile
        current_profile = get_user_profile(user_id)
        
        # If no profile exists, or profile is minimal, extract one
        if not current_profile or (not current_profile.name and len(current_profile.interests) < 2):
            # Extract profile from conversation
            new_profile = extract_profile_from_conversation(model, conversation_history)
            
            # Merge with existing profile if there is one
            if current_profile:
                # Keep non-empty values from current profile
                for field, value in current_profile.model_dump().items():
                    if value and not getattr(new_profile, field):
                        setattr(new_profile, field, value)
            
            # Store updated profile
            store_user_profile(user_id, new_profile)
            return True
        
        return False
    except Exception as e:
        logging.error(f"Error updating profile: {str(e)}")
        return False

def chat_with_agent(chat_request: ChatRequest) -> ChatResponse:
    """Process a chat request through the agent."""
    try:
        user_id = chat_request.user_id
        user_message = chat_request.message
        
        # Get user profile
        profile = get_user_profile(user_id)
        profile_text = profile.model_dump_json() if profile else "No profile available yet"
        
        # Get recent conversations
        recent_convs = get_recent_conversations(user_id)
        conversations_text = "\n".join([
            f"[{c.timestamp.isoformat()}]\nUser: {c.user_message}\nAssistant: {c.agent_response}"
            for c in recent_convs
        ])
        
        # Create message history
        history = []
        for conv in recent_convs:
            history.append(HumanMessage(content=conv.user_message))
            history.append(AIMessage(content=conv.agent_response))
        
        # Add the current message
        history.append(HumanMessage(content=user_message))
        
        # Configure agent with user context
        config = RunnableConfig(
            configurable={
                "user_id": str(user_id)
            }
        )
        
        # Process with agent graph
        result = agent_graph.invoke(
            {"messages": history},
            config=config
        )
        
        # Get the final message
        final_message = result["messages"][-1]
        agent_response = final_message.content
        
        # Store the conversation
        store_conversation(user_id, user_message, agent_response)
        
        # Update profile if needed
        updated_profile = update_profile_if_needed(
            user_id, 
            history + [AIMessage(content=agent_response)]
        )
        
        return ChatResponse(
            response=agent_response,
            updated_profile=updated_profile
        )
    except Exception as e:
        logging.error(f"Error in chat_with_agent: {str(e)}")
        return ChatResponse(
            response=f"I'm sorry, I encountered an error while processing your message. Please try again later.",
            updated_profile=False
        )