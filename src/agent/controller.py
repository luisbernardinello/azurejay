from fastapi import APIRouter, HTTPException, status

from ..database.core import DbSession
from ..auth.service import CurrentUser
from . import models
from . import service

router = APIRouter(
    prefix="/agent",
    tags=["Agent"]
)

@router.post("/chat", response_model=models.ChatResponse)
async def chat_with_agent(
    chat_request: models.ChatRequest,
    current_user: CurrentUser,
    db: DbSession
):
    """Chat with the AI agent."""
    # Set the user_id from the authenticated user
    chat_request.user_id = current_user.get_uuid()

    # Process the chat request
    try:
        return service.chat_with_agent(chat_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

@router.get("/profile", response_model=models.Profile)
async def get_profile(
    current_user: CurrentUser,
    db: DbSession
):
    """Get the user's profile from the agent memory."""
    try:
        profile = service.get_user_profile(current_user.get_uuid())
        if not profile:
            return models.Profile()
        return profile
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving profile: {str(e)}"
        )