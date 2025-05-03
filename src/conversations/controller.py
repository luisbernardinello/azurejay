from fastapi import APIRouter, status, HTTPException
from uuid import UUID
from typing import List

from src.database.core import RedisClient
from src.auth.service import CurrentUser
from . import models
from . import service

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=UUID)
def create_conversation(
    conversation_data: models.ConversationCreate,
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Cria uma nova conversa para o usuário
    """
    return service.create_conversation(
        redis_client, 
        current_user.get_uuid(), 
        conversation_data
    )


@router.get("/", response_model=List[models.ConversationResponse])
def get_user_conversations(
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Retorna todas as conversas do usuário
    """
    return service.get_user_conversations(redis_client, current_user.get_uuid())


@router.get("/{conversation_id}", response_model=models.ConversationDetailResponse)
def get_conversation(
    conversation_id: UUID,
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Retorna os detalhes de uma conversa específica
    """
    return service.get_conversation(
        redis_client, 
        current_user.get_uuid(), 
        conversation_id
    )


@router.post("/{conversation_id}/messages", response_model=models.MessageResponse)
def add_message(
    conversation_id: UUID,
    message_data: models.MessageRequest,
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Adiciona uma nova mensagem do usuário à conversa
    """
    return service.add_message(
        redis_client, 
        current_user.get_uuid(), 
        conversation_id, 
        message_data
    )


@router.post("/{conversation_id}/system-messages", response_model=models.MessageResponse)
def add_system_message(
    conversation_id: UUID,
    message_data: models.MessageRequest,
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Adiciona uma nova mensagem do sistema à conversa
    """
    return service.add_message(
        redis_client, 
        current_user.get_uuid(), 
        conversation_id, 
        message_data,
        is_user=False
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: UUID,
    redis_client: RedisClient,
    current_user: CurrentUser
):
    """
    Exclui uma conversa
    """
    service.delete_conversation(
        redis_client, 
        current_user.get_uuid(), 
        conversation_id
    )