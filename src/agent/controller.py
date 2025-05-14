from fastapi import APIRouter, status, HTTPException, Depends, BackgroundTasks
from uuid import UUID
from typing import List
from sqlalchemy.orm import Session

from src.database.core import RedisClient, get_db
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
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
    redis_client: RedisClient,
    db: Session = Depends(get_db)
):
    """
    Process a message through the English tutor agent and store the conversation
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
                conversation_service.models.ConversationCreate(title="English Tutor Chat")
            )
        
        # Process the message through the agent
        response = service.process_message(
            redis_client,
            db,
            current_user.get_uuid(),
            conversation_id,
            agent_request.message
        )
        
        # Store the user message in the conversation
        message_request = conversation_service.models.MessageRequest(content=agent_request.message)
        conversation_service.add_message(redis_client, current_user.get_uuid(), conversation_id, message_request, is_user=True)
        
        # Store the AI response in the conversation
        ai_message_request = conversation_service.models.MessageRequest(content=response.message)
        conversation_service.add_message(redis_client, current_user.get_uuid(), conversation_id, ai_message_request, is_user=False)
        
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
    Get the agent's memory for the current user, including profile, topics, grammar corrections, and web search results
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
    Reset (delete) the agent's memory for the current user, including profile, topics, grammar corrections, and web search results
    """
    try:
        service.reset_user_memory(redis_client, current_user.get_uuid())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset memory: {str(e)}"
        )


@router.get("/grammar", response_model=List[models.GrammarCorrection])
async def get_grammar_corrections(
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Get all grammar corrections for the current user
    """
    try:
        memory = service.get_user_memory(redis_client, current_user.get_uuid())
        grammar_corrections = memory.memory.grammar_corrections if memory.memory else []
        return grammar_corrections
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get grammar corrections: {str(e)}"
        )


@router.get("/topics", response_model=List[models.ConversationTopic])
async def get_topics(
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Get all conversation topics for the current user
    """
    try:
        memory = service.get_user_memory(redis_client, current_user.get_uuid())
        topics = memory.memory.topics if memory.memory else []
        return topics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get topics: {str(e)}"
        )