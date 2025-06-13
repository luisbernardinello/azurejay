# src/conversations/service.py
import json
import logging
from uuid import UUID, uuid4
from datetime import datetime
from typing import List
from time import time
import traceback

import redis
from langgraph.checkpoint.base import Checkpoint
from langchain_core.messages import HumanMessage, AIMessage

from . import models
from ..agent.service import get_agent_graph

def get_user_conversations_list(
    redis_client: redis.Redis,
    user_id: UUID
) -> List[models.ConversationListItem]:
    """
    Gets a user's conversation list from the Redis ZSET index.
    Returns a list sorted by the most recently updated.
    """
    index_key = f"user_conversations:{user_id}"
    try:
        # Fetch all conversations from the ZSET, from newest to oldest.
        # The result is a list of tuples: (member, score)
        conv_data = redis_client.zrange(index_key, 0, -1, desc=True, withscores=True)

        conversations = []
        for item_json, score in conv_data:
            item_data = json.loads(item_json)
            conversations.append(models.ConversationListItem(
                id=UUID(item_data['id']),
                title=item_data['title'],
                # The score is stored as a Unix timestamp.
                updated_at=datetime.fromtimestamp(float(score))
            ))
        return conversations
    except Exception as e:
        logging.error(f"Error fetching conversation list for user {user_id}: {e}")
        return []

def get_conversation_history(
    checkpointer,  # Will be CheckpointerDep
    redis_client: redis.Redis, # Will be RedisDep
    user_id: UUID,
    conversation_id: UUID
) -> models.ConversationHistoryResponse:
    """
    Retrieves the full message history for a specific conversation.
    """
    config = {"configurable": {"thread_id": str(conversation_id)}}
    # We retrieve the latest checkpoint for the conversation thread.
    checkpoint = checkpointer.get(config)

    if not checkpoint:
        raise FileNotFoundError(f"Conversation with ID {conversation_id} not found.")

    # --- Security Check ---
    # Ensure the requesting user is the owner of the conversation.
    checkpoint_user_id = checkpoint.get("config", {}).get("configurable", {}).get("user_id")
    if str(user_id) != checkpoint_user_id:
        logging.warning(f"User {user_id} attempted to access conversation {conversation_id} owned by {checkpoint_user_id}.")
        raise PermissionError("Access denied: You do not own this conversation.")

    # --- Data Transformation ---
    # Transform the raw LangGraph messages into a clean, frontend-friendly format.
    messages = []
    raw_messages = checkpoint.get("channel_values", {}).get("messages", [])

    for i, msg in enumerate(raw_messages):
        # Skip internal, non-content messages.
        if msg.name in ["supervisor", "enhancer"]:
            continue
        
        # Process human messages and look for subsequent analysis.
        if msg.type == 'human' and (not hasattr(msg, 'name') or msg.name is None):
            analysis_content = None
            # Check if the next message is a correction from the correction_node.
            if (i + 1) < len(raw_messages) and raw_messages[i+1].name == "correction":
                correction_msg = raw_messages[i+1].content
                if correction_msg != "CORRECT":
                    # Attach the correction details.
                    analysis_content = {"correction": correction_msg}

            messages.append(models.MessageDetail(
                role='human',
                content=msg.content,
                analysis=analysis_content
            ))
        # Process AI messages
        elif msg.type == 'ai':
             messages.append(models.MessageDetail(
                role='ai',
                content=msg.content,
            ))

    # Retrieve the title from our Redis index for consistency.
    index_key = f"user_conversations:{user_id}"
    title = f"Conversation {conversation_id}" # Default title
    items_json = redis_client.zrange(index_key, 0, -1)
    for item_json in items_json:
        item_data = json.loads(item_json)
        if item_data['id'] == str(conversation_id):
            title = item_data['title']
            break

    return models.ConversationHistoryResponse(
        id=conversation_id,
        title=title,
        messages=messages
    )

async def create_new_conversation(
    redis_client: redis.Redis,
    user_id: UUID,
    request: models.NewConversationRequest,
) -> models.NewConversationResponse:
    """
    Creates a new conversation with the user's first message.
    This function handles the entire flow for the /new endpoint.
    """
    # Generate new conversation ID
    conversation_id = uuid4()
    
    # Create title from first message (first 60 characters)
    title = request.content[:60].strip()
    if len(request.content) > 60:
        title += "..."
    
    config = {
        "configurable": {
            "thread_id": str(conversation_id),
            "user_id": str(user_id),
        }
    }

    app = get_agent_graph()
    input_messages = [HumanMessage(content=request.content)]
    final_response = None

    try:
        # Collect all messages during streaming
        all_messages = []
        
        # Asynchronously stream the graph's execution
        async for chunk in app.astream(
            {"messages": input_messages}, config, stream_mode="values"
        ):
            if "messages" in chunk and chunk["messages"]:
                all_messages = chunk["messages"]  # Keep the latest complete version
                
                # Get the last message from the chunk
                last_message = chunk["messages"][-1]
                
                # DEBUG: Log message details to understand structure
                logging.info(f"DEBUG - Message type: {type(last_message)}")
                logging.info(f"DEBUG - Message content: {last_message.content[:100] if hasattr(last_message, 'content') else 'No content'}")
                logging.info(f"DEBUG - Has name attribute: {hasattr(last_message, 'name')}")
                if hasattr(last_message, 'name'):
                    logging.info(f"DEBUG - Name value: {last_message.name}")
                
                # CORRECTION: Identify final response from responder
                # Can be AIMessage without name OR with name None
                if (isinstance(last_message, AIMessage) and 
                    (not hasattr(last_message, 'name') or last_message.name is None) and 
                    last_message.content.strip()):
                    final_response = last_message
                    logging.info(f"Final response captured from responder: {last_message.content[:100]}...")
                    break  # Important: exit loop once final response is found
                
                # ALTERNATIVE: If above condition doesn't work, try to capture 
                # the last AI message regardless of name (as fallback)
                elif isinstance(last_message, AIMessage) and last_message.content.strip():
                    # Check if it's not from a known agent
                    if not (hasattr(last_message, 'name') and 
                           last_message.name in ['supervisor', 'correction', 'researcher', 'enhancer', 'validator']):
                        final_response = last_message
                        logging.info(f"Final response captured (fallback): {last_message.content[:100]}...")
        
        # If not found during streaming, search in final list
        if final_response is None and all_messages:
            logging.info("Searching for final response in all messages...")
            # Search backwards for a valid AI message
            for msg in reversed(all_messages):
                if (isinstance(msg, AIMessage) and 
                    msg.content.strip() and 
                    not (hasattr(msg, 'name') and 
                         msg.name in ['supervisor', 'correction', 'researcher', 'enhancer', 'validator'])):
                    final_response = msg
                    logging.info(f"Final response found in message history: {msg.content[:100]}...")
                    break

        if final_response is None:
            raise Exception("Agent did not produce a final response.")

        # --- CREATE CONVERSATION INDEX IN REDIS ---
        index_key = f"user_conversations:{user_id}"
        timestamp = time()
        
        # Create new conversation entry
        conv_meta = json.dumps({"id": str(conversation_id), "title": title})
        redis_client.zadd(index_key, {conv_meta: timestamp})
        logging.info(f"Created new conversation {conversation_id} in index for user {user_id}.")

        return models.NewConversationResponse(
            response=final_response.content,
            conversation_id=conversation_id,
            title=title
        )

    except Exception as e:
        logging.error(
            f"Error during new conversation creation for user {user_id}: {e}"
        )
        logging.error(traceback.format_exc())
        raise