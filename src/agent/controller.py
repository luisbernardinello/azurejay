# src/agent/controller.py
from fastapi import APIRouter, status, Body, HTTPException, Path
from uuid import UUID

from src.database.core import DbSession, CheckpointerDep, StoreDep
from src.auth.service import CurrentUser
from . import models
from . import service
from src.conversations.service import add_message_to_conversation

router = APIRouter(
    prefix="/agent",
    tags=["Agent"]
)

@router.post(
    "/chat/{conversation_id}",
    response_model=models.AgentResponse,
    status_code=status.HTTP_200_OK
)
async def chat_with_existing_conversation(
    conversation_id: UUID = Path(..., description="The ID of the existing conversation"),
    current_user: CurrentUser = None,
    db: DbSession = None,
    request: models.AgentChatRequest = Body(...)
):
    """
    Endpoint for continuing a conversation with the AI tutor.
    This endpoint is used when the user is on /chat/{id} route.
    Uses PostgreSQL for conversation storage and Redis only for agent state.
    """
    user_id = current_user.get_uuid()
    
    try:
        # Validate that the conversation exists and belongs to the user
        await service.validate_conversation_access(db, user_id, conversation_id)
        
        # Create the agent request with the existing conversation ID
        agent_request = models.AgentRequest(
            content=request.content,
            conversation_id=conversation_id
        )
        
        return await service.chat_with_agent(
            db=db,
            user_id=user_id,
            request=agent_request,
            add_message_func=add_message_to_conversation
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred in the agent service: {str(e)}"
        )