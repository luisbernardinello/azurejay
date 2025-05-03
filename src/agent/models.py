from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class Profile(BaseModel):
    """This is the profile of the user."""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    occupation: Optional[str] = Field(description="The user's job or occupation", default=None)
    interests: List[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )
    preferences: List[str] = Field(
        description="User preferences for conversation topics",
        default_factory=list
    )

class Conversation(BaseModel):
    """A record of a conversation with the agent."""
    timestamp: datetime = Field(default_factory=datetime.now)
    user_message: str = Field(description="Message from the user")
    agent_response: str = Field(description="Response from the agent")

class ChatRequest(BaseModel):
    """Request for chatting with the agent."""
    message: str
    user_id: UUID

class ChatResponse(BaseModel):
    """Response from the agent."""
    response: str
    updated_profile: bool = False