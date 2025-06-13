# src/agent/service.py
import json
import logging
from time import time
from uuid import UUID, uuid4
import traceback
from typing import Callable, Awaitable
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage

from . import models
from .supervisor import create_agent_graph
from ..entities.conversation import Conversation
from ..exceptions import ConversationNotFoundError
# REMOVIDO: from ..conversations.service import add_message_to_conversation

# A global variable to hold the compiled graph singleton
agent_graph = None

def get_agent_graph(): # type: ignore
    """
    Creates and returns a singleton instance of the compiled agent graph.
    This ensures the graph is only compiled once per application lifecycle.
    """
    global agent_graph
    if agent_graph is None:
        logging.info("Compiling agent graph for the first time.")
        agent_graph = create_agent_graph()
    return agent_graph

async def validate_conversation_access(
    db: Session,
    user_id: UUID,
    conversation_id: UUID
) -> None:
    """
    Validates that a conversation exists and belongs to the specified user in PostgreSQL.
    Raises appropriate exceptions if validation fails.
    """
    try:
        # Check if the conversation exists and belongs to the user
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
            .first()
        )
        
        if not conversation:
            raise ConversationNotFoundError(conversation_id)
            
    except ConversationNotFoundError:
        raise
    except Exception as e:
        logging.error(f"Error validating conversation access for user {user_id}, conversation {conversation_id}: {e}")
        raise

async def chat_with_agent(
    db: Session,
    user_id: UUID,
    request: models.AgentRequest,
    add_message_func: Callable[[Session, UUID, UUID, str, str, dict], None]  # UPDATED: Added dict parameter for improvement_analysis
) -> models.AgentResponse:
    """
    Processes a chat message with the AI agent and updates the conversation in PostgreSQL.
    This function is used for continuing existing conversations (when conversation_id is provided).
    
    Args:
        db: Database session
        user_id: User ID
        request: Agent request
        add_message_func: Function to add messages to conversation (injected dependency)
    """
    if not request.conversation_id:
        raise ValueError("conversation_id is required for chat_with_agent. Use create_new_conversation for new conversations.")
    
    conversation_id = request.conversation_id
    
    # Validate conversation access first
    await validate_conversation_access(db, user_id, conversation_id)
    
    config = {
        "configurable": {
            "thread_id": str(conversation_id),
            "user_id": str(user_id),
        }
    }

    app = get_agent_graph()
    input_messages = [HumanMessage(content=request.content)]
    final_response = None
    improvement_analysis = None

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
                    
                    # Check if this message has improvement analysis
                    if hasattr(last_message, 'additional_kwargs') and 'improvement' in last_message.additional_kwargs:
                        improvement_analysis = {"improvement": last_message.additional_kwargs['improvement']}
                        logging.info(f"Improvement analysis captured: {improvement_analysis}")
                    
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
                    
                    # Check if this message has improvement analysis
                    if hasattr(msg, 'additional_kwargs') and 'improvement' in msg.additional_kwargs:
                        improvement_analysis = {"improvement": msg.additional_kwargs['improvement']}
                        logging.info(f"Improvement analysis found in message history: {improvement_analysis}")
                    
                    break

        if final_response is None:
            raise Exception("Agent did not produce a final response.")

        # --- ADD MESSAGES TO CONVERSATION USING INJECTED FUNCTION ---
        add_message_func(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            human_message=request.content,
            ai_response=final_response.content,
            improvement_analysis=improvement_analysis  # Pass the improvement analysis
        )

        return models.AgentResponse(
            response=final_response.content,
            conversation_id=conversation_id,
        )

    except Exception as e:
        logging.error(
            f"Error during agent execution for user {user_id} in conversation {conversation_id}: {e}"
        )
        # Re-raise the exception to be handled by the controller
        logging.error(traceback.format_exc())
        raise