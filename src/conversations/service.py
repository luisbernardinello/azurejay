# src/conversations/service.py
import json
import logging
from uuid import UUID, uuid4
from datetime import datetime
from typing import List
import traceback

from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage

from . import models
from ..entities.conversation import Conversation
from ..agent.service import get_agent_graph
from ..exceptions import ConversationNotFoundError


def get_user_conversations_list(
    db: Session,
    user_id: UUID
) -> List[models.ConversationListItem]:
    """
    Gets a user's conversation list from PostgreSQL.
    Returns a list sorted by the most recently updated.
    """
    try:
        conversations = (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )
        
        return [
            models.ConversationListItem(
                id=conv.id,
                title=conv.title,
                updated_at=conv.updated_at
            )
            for conv in conversations
        ]
    except Exception as e:
        logging.error(f"Error fetching conversation list for user {user_id}: {e}")
        return []


def get_conversation_history(
    db: Session,
    checkpointer,  # Will be CheckpointerDep
    user_id: UUID,
    conversation_id: UUID
) -> models.ConversationHistoryResponse:
    """
    Retrieves the full message history for a specific conversation from PostgreSQL.
    """
    try:
        # Buscar a conversa no PostgreSQL
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
        
        # Converter as mensagens do formato JSON para MessageDetail
        messages = []
        for msg_data in conversation.messages:
            messages.append(models.MessageDetail(
                role=msg_data['role'],
                content=msg_data['content'],
                analysis=msg_data.get('analysis')
            ))
        
        return models.ConversationHistoryResponse(
            id=conversation.id,
            title=conversation.title,
            messages=messages
        )
        
    except ConversationNotFoundError:
        raise
    except Exception as e:
        logging.error(f"Error fetching conversation history for {conversation_id}: {e}")
        raise


async def create_new_conversation(
    db: Session,
    user_id: UUID,
    request: models.NewConversationRequest,
) -> models.NewConversationResponse:
    """
    Creates a new conversation with the user's first message in PostgreSQL.
    Uses LangGraph only for agent processing, stores conversation in DB.
    """
    # Generate new conversation ID
    conversation_id = uuid4()
    
    # Create title from first message (first 60 characters)
    title = request.content[:60].strip()
    if len(request.content) > 60:
        title += "..."
    
    # Config for LangGraph (Redis will only store the agent state)
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

        # --- CREATE CONVERSATION IN POSTGRESQL ---
        # Preparar mensagens para armazenar no JSON
        messages_to_store = [
            {
                "role": "human",
                "content": request.content,
                "analysis": None  # Você pode adicionar análise aqui se necessário
            },
            {
                "role": "ai",
                "content": final_response.content,
                "analysis": None
            }
        ]
        
        # Criar nova conversa no PostgreSQL
        new_conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title,
            messages=messages_to_store,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        
        logging.info(f"Created new conversation {conversation_id} in PostgreSQL for user {user_id}.")

        return models.NewConversationResponse(
            response=final_response.content,
            conversation_id=conversation_id,
            title=title
        )

    except Exception as e:
        db.rollback()
        logging.error(
            f"Error during new conversation creation for user {user_id}: {e}"
        )
        logging.error(traceback.format_exc())
        raise


def add_message_to_conversation(
    db: Session,
    user_id: UUID,
    conversation_id: UUID,
    human_message: str,
    ai_response: str
) -> None:
    """
    Adds new messages to an existing conversation in PostgreSQL.
    """
    try:
        # Buscar a conversa
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
        
        # Adicionar novas mensagens ao array JSON
        new_messages = [
            {
                "role": "human",
                "content": human_message,
                "analysis": None
            },
            {
                "role": "ai",
                "content": ai_response,
                "analysis": None
            }
        ]
        
        # Atualizar mensagens e timestamp
        conversation.messages.extend(new_messages)
        conversation.updated_at = datetime.utcnow()
        
        # Marcar o campo como modificado para o SQLAlchemy
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(conversation, 'messages')
        
        db.commit()
        logging.info(f"Added messages to conversation {conversation_id} for user {user_id}.")
        
    except ConversationNotFoundError:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"Error adding message to conversation {conversation_id}: {e}")
        raise