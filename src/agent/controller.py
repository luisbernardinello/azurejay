from fastapi import APIRouter, status, HTTPException, Depends, BackgroundTasks
from uuid import UUID
from typing import List

from src.database.core import RedisClient
from src.auth.service import CurrentUser
from . import models
from . import service
from ..conversations import service as conversation_service

router = APIRouter(
    prefix="/agent",
    tags=["Agent"]
)


@router.post("/chat/{conversation_id}", response_model=models.AgentResponse)
async def process_message(
    conversation_id: UUID,
    agent_request: models.AgentRequest,
    redis_client: RedisClient,
    current_user: CurrentUser,
    background_tasks: BackgroundTasks
):
    """
    Process a message through the agent and store the conversation
    """
    try:
        # Verify the conversation exists and belongs to the user
        try:
            conversation_service.get_conversation(redis_client, current_user.get_uuid(), conversation_id)
        except Exception:
            # Create a new conversation if it doesn't exist
            conversation_service.create_conversation(
                redis_client,
                current_user.get_uuid(),
                conversation_service.models.ConversationCreate(title="AI Assistant Chat")
            )
        
        # Process the message through the agent (this can be slow, consider moving to background)
        response = service.process_message(
            redis_client,
            current_user.get_uuid(),
            conversation_id,
            agent_request.message
        )
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@router.get("/memory", response_model=models.AgentMemoryResponse)
async def get_memory(
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Get the agent's memory for the current user
    """
    try:
        memory = service.get_user_memory(redis_client, current_user.get_uuid())
        return memory
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory: {str(e)}"
        )


@router.delete("/memory", status_code=status.HTTP_204_NO_CONTENT)
async def reset_memory(
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Reset (delete) the agent's memory for the current user
    """
    try:
        service.reset_user_memory(redis_client, current_user.get_uuid())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset memory: {str(e)}"
        )