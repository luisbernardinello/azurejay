from pydantic import BaseModel, Field
from uuid import UUID
from typing import List, Dict, Any, Optional


class UserProfileMemory(BaseModel):
    """Profile of a user that's extracted and maintained by the agent"""
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location") 
    user_interests: List[str] = Field(description="A list of the user's interests", default_factory=list)
    
    
class AgentConfig(BaseModel):
    """Configuration for the agent"""
    user_id: UUID
    conversation_id: UUID
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0


class AgentRequest(BaseModel):
    """Request to process a message with the agent"""
    message: str


class AgentResponse(BaseModel):
    """Response from the agent"""
    message: str
    updated_memory: Optional[UserProfileMemory] = None


class AgentMemoryResponse(BaseModel):
    """Response containing the agent's memory for a user"""
    user_id: UUID
    memory: Optional[UserProfileMemory] = None
    