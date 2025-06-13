# src/conversations/models.py
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import List, Literal, Optional

class ConversationListItem(BaseModel):
    """
    Represents a single item in the user's conversation list (for the side menu).
    """
    id: UUID
    title: str
    updated_at: datetime

class MessageDetail(BaseModel):
    """
    Represents a single, detailed message within a conversation's history.
    """
    role: Literal['human', 'ai']
    content: str
    # Optional field to hold analysis like grammar corrections for the frontend.
    analysis: Optional[dict] = None

class ConversationHistoryResponse(BaseModel):
    """
    Represents the full history of a specific conversation.
    """
    id: UUID
    title: str
    messages: List[MessageDetail]

# Novos modelos para a funcionalidade /new
class NewConversationRequest(BaseModel):
    """
    Defines the structure for creating a new conversation with the first message.
    """
    content: str = Field(
        ...,
        description="The first message content from the user.",
        examples=["Hello, my name is John and I like to play soccer."]
    )

class NewConversationResponse(BaseModel):
    """
    Defines the response structure when creating a new conversation.
    Contains the AI response and the new conversation ID for frontend redirection.
    """
    response: str = Field(
        ...,
        description="The AI response to the user's first message."
    )
    conversation_id: UUID = Field(
        ...,
        description="The ID of the newly created conversation for frontend redirection."
    )
    title: str = Field(
        ...,
        description="The title of the new conversation (derived from first message)."
    )