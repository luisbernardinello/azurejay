from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import List, Optional


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: UUID
    content: str
    timestamp: datetime
    is_user: bool


class ConversationCreate(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationDetailResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]