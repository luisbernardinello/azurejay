# src/conversations/controller.py
from fastapi import APIRouter, HTTPException, status
from uuid import UUID
from typing import List
import logging

from src.database.core import DbSession, CheckpointerDep
from src.auth.service import CurrentUser
from . import models, service

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)

@router.get("/", response_model=List[models.ConversationListItem])
def list_user_conversations(
    db: DbSession,
    current_user: CurrentUser
):
    """
    Returns the list of conversations for the authenticated user from PostgreSQL,
    sorted by the most recently updated.
    """
    user_id = current_user.get_uuid()
    return service.get_user_conversations_list(db, user_id)

@router.get("/{conversation_id}", response_model=models.ConversationHistoryResponse)
def get_conversation_details(
    conversation_id: UUID,
    db: DbSession,
    checkpointer: CheckpointerDep,
    current_user: CurrentUser
):
    """
    Returns the detailed message history for a specific conversation from PostgreSQL.
    Ensures the user has permission to view the conversation.
    """
    user_id = current_user.get_uuid()
    try:
        return service.get_conversation_history(db, checkpointer, user_id, conversation_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching conversation {conversation_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.")

@router.post("/new", response_model=models.NewConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_new_conversation(
    current_user: CurrentUser,
    db: DbSession,
    request: models.NewConversationRequest
):
    """
    Creates a new conversation with the first message from the user in PostgreSQL.
    This endpoint is called when the user is on /new route and sends their first message.
    Returns the AI response and the new conversation ID for frontend redirection.
    """
    user_id = current_user.get_uuid()
    try:
        return await service.create_new_conversation(
            db=db,
            user_id=user_id,
            request=request
        )
    except Exception as e:
        logging.error(f"Error creating new conversation for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while creating the conversation: {str(e)}"
        )