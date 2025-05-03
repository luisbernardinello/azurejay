import json
import logging
from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Optional

import redis

from src.entities.conversation import Conversation, Message
from src.exceptions import ConversationNotFoundError
from . import models


def get_conversation_key(conversation_id: UUID) -> str:
    """
    Gera a chave Redis para uma conversa específica
    """
    return f"conversation:{conversation_id}"


def get_user_conversations_key(user_id: UUID) -> str:
    """
    Gera a chave Redis para o conjunto de conversas de um usuário
    """
    return f"user:{user_id}:conversations"


def create_conversation(
    redis_client: redis.Redis, 
    user_id: UUID, 
    conversation_data: models.ConversationCreate
) -> UUID:
    """
    Cria uma nova conversa para o usuário
    """
    try:
        conversation_id = uuid4()
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=conversation_data.title
        )
        
        # Salvar a conversa no Redis
        redis_client.set(
            get_conversation_key(conversation_id), 
            conversation.model_dump_json()
        )
        
        # Adicionar a conversa ao conjunto de conversas do usuário
        redis_client.sadd(
            get_user_conversations_key(user_id),
            str(conversation_id)
        )
        
        logging.info(f"Created new conversation {conversation_id} for user {user_id}")
        return conversation_id
    except Exception as e:
        logging.error(f"Error creating conversation for user {user_id}: {str(e)}")
        raise


def get_user_conversations(
    redis_client: redis.Redis, 
    user_id: UUID
) -> List[models.ConversationResponse]:
    """
    Retorna todas as conversas de um usuário
    """
    try:
        # Obter os IDs de todas as conversas do usuário
        conversation_ids = redis_client.smembers(get_user_conversations_key(user_id))
        
        conversations = []
        for conv_id in conversation_ids:
            # Obter cada conversa do Redis
            conv_data = redis_client.get(get_conversation_key(conv_id))
            if conv_data:
                conversation = Conversation.from_json(conv_data)
                conversations.append(models.ConversationResponse(
                    id=conversation.id,
                    title=conversation.title,
                    created_at=conversation.created_at,
                    updated_at=conversation.updated_at,
                    message_count=len(conversation.messages)
                ))
        
        # Ordenar por data de atualização, mais recente primeiro
        conversations.sort(key=lambda x: x.updated_at, reverse=True)
        return conversations
    except Exception as e:
        logging.error(f"Error fetching conversations for user {user_id}: {str(e)}")
        raise


def get_conversation(
    redis_client: redis.Redis, 
    user_id: UUID, 
    conversation_id: UUID
) -> models.ConversationDetailResponse:
    """
    Retorna os detalhes de uma conversa específica
    """
    try:
        # Verificar se a conversa existe para o usuário
        if not redis_client.sismember(get_user_conversations_key(user_id), str(conversation_id)):
            raise ConversationNotFoundError(conversation_id)
            
        # Obter a conversa do Redis
        conv_data = redis_client.get(get_conversation_key(conversation_id))
        if not conv_data:
            raise ConversationNotFoundError(conversation_id)
            
        conversation = Conversation.from_json(conv_data)
        
        # Verificar se a conversa pertence ao usuário
        if conversation.user_id != user_id:
            logging.warning(f"User {user_id} attempted to access conversation {conversation_id} belonging to another user")
            raise ConversationNotFoundError(conversation_id)
            
        # Converter para o modelo de resposta
        messages = [
            models.MessageResponse(
                id=msg.id,
                content=msg.content,
                timestamp=msg.timestamp,
                is_user=msg.is_user
            ) for msg in conversation.messages
        ]
        
        return models.ConversationDetailResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages
        )
    except ConversationNotFoundError:
        raise
    except Exception as e:
        logging.error(f"Error fetching conversation {conversation_id} for user {user_id}: {str(e)}")
        raise


def add_message(
    redis_client: redis.Redis, 
    user_id: UUID, 
    conversation_id: UUID, 
    message_data: models.MessageRequest,
    is_user: bool = True
) -> models.MessageResponse:
    """
    Adiciona uma nova mensagem a uma conversa
    """
    try:
        # Verificar se a conversa existe
        conv_data = redis_client.get(get_conversation_key(conversation_id))
        if not conv_data:
            raise ConversationNotFoundError(conversation_id)
            
        conversation = Conversation.from_json(conv_data)
        
        # Verificar se a conversa pertence ao usuário
        if conversation.user_id != user_id:
            logging.warning(f"User {user_id} attempted to add message to conversation {conversation_id} belonging to another user")
            raise ConversationNotFoundError(conversation_id)
            
        # Adicionar a mensagem
        message = conversation.add_message(message_data.content, is_user)
        
        # Atualizar a conversa no Redis
        redis_client.set(
            get_conversation_key(conversation_id),
            conversation.model_dump_json()
        )
        
        return models.MessageResponse(
            id=message.id,
            content=message.content,
            timestamp=message.timestamp,
            is_user=message.is_user
        )
    except ConversationNotFoundError:
        raise
    except Exception as e:
        logging.error(f"Error adding message to conversation {conversation_id} for user {user_id}: {str(e)}")
        raise


def delete_conversation(
    redis_client: redis.Redis, 
    user_id: UUID, 
    conversation_id: UUID
) -> None:
    """
    Exclui uma conversa
    """
    try:
        # Verificar se a conversa existe para o usuário
        if not redis_client.sismember(get_user_conversations_key(user_id), str(conversation_id)):
            raise ConversationNotFoundError(conversation_id)
            
        # Obter a conversa do Redis para verificar o proprietário
        conv_data = redis_client.get(get_conversation_key(conversation_id))
        if not conv_data:
            raise ConversationNotFoundError(conversation_id)
            
        conversation = Conversation.from_json(conv_data)
        
        # Verificar se a conversa pertence ao usuário
        if conversation.user_id != user_id:
            logging.warning(f"User {user_id} attempted to delete conversation {conversation_id} belonging to another user")
            raise ConversationNotFoundError(conversation_id)
            
        # Excluir a conversa do Redis
        redis_client.delete(get_conversation_key(conversation_id))
        
        # Remover a conversa do conjunto de conversas do usuário
        redis_client.srem(get_user_conversations_key(user_id), str(conversation_id))
        
        logging.info(f"Deleted conversation {conversation_id} for user {user_id}")
    except ConversationNotFoundError:
        raise
    except Exception as e:
        logging.error(f"Error deleting conversation {conversation_id} for user {user_id}: {str(e)}")
        raise