from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Optional


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    is_user: bool  # True se a mensagem for do usuário, False se for do sistema


class Conversation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    title: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, content: str, is_user: bool = True) -> Message:
        """
        Adiciona uma nova mensagem à conversa e atualiza o timestamp
        """
        message = Message(
            content=content,
            is_user=is_user
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def model_dump_json(self) -> str:
        """
        Converte a conversa para JSON para armazenamento
        """
        return super().model_dump_json()
    
    @classmethod
    def from_json(cls, json_data: str) -> "Conversation":
        """
        Cria uma instância de Conversation a partir de dados JSON
        """
        return cls.model_validate_json(json_data)