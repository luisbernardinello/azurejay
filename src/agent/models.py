# src/agent/models.py
from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional

class AgentRequest(BaseModel):
    """
    Defines the structure for a user's request to the agent.
    This is used internally by the service layer.
    """
    content: str = Field(
        ...,
        description="The content of the user's message.",
        examples=["Hello, my name is John and I like to play soccer."]
    )
    conversation_id: Optional[UUID] = Field(
        default=None,
        description="The ID of an existing conversation. If None, a new conversation will be started."
    )

class AgentChatRequest(BaseModel):
    """
    Defines the structure for a user's request when chatting in an existing conversation.
    Used by the /chat/{id} endpoint.
    """
    content: str = Field(
        ...,
        description="The content of the user's message.",
        examples=["Can you help me practice my English?"]
    )

class AgentResponse(BaseModel):
    """
    Defines the structure of the agent's response.
    """
    response: str = Field(
        ...,
        description="The text response generated by the agent."
    )
    conversation_id: UUID = Field(
        ...,
        description="The ID of the conversation, used to continue the chat in subsequent requests."
    )