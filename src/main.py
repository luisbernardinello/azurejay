from fastapi import FastAPI
from .database.core import Base, engine, verify_database_connections
from .entities.user import User  # Import models to register them
from .entities.conversation import Conversation  # Import conversation model
from .api import register_routes
from .logging import configure_logging, LogLevels
from sqladmin import Admin, ModelView

configure_logging(LogLevels.info)

app = FastAPI()

# Configurar o SQLAdmin
admin = Admin(app, engine)

# Criar classes de administração para cada modelo
class UserAdmin(ModelView, model=User):
    column_list = [User.id, User.email, User.first_name, User.last_name]
    column_searchable_list = [User.email, User.first_name, User.last_name]
    column_sortable_list = [User.id, User.email, User.first_name, User.last_name]
    column_default_sort = [(User.id, True)]  # Ordenar por ID de forma descendente
    name = "Usuário"
    name_plural = "Usuários"
    icon = "fa-solid fa-user"
    
    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True

class ConversationAdmin(ModelView, model=Conversation):
    column_list = [Conversation.id, Conversation.title, Conversation.user_id, Conversation.created_at, Conversation.updated_at]
    column_searchable_list = [Conversation.title]
    column_sortable_list = [Conversation.created_at, Conversation.updated_at, Conversation.title]
    column_default_sort = [(Conversation.updated_at, True)]  # Ordenar por data de atualização
    name = "Conversa"
    name_plural = "Conversas"
    icon = "fa-solid fa-comments"
    
    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True

# Registrar modelos no admin
admin.add_view(UserAdmin)
admin.add_view(ConversationAdmin)

@app.on_event("startup")
async def startup_db_client():
    # Verificar todas as conexões de banco de dados na inicialização
    verify_database_connections()
    # Descomentar a linha abaixo para apagar as tabelas existentes
    # Base.metadata.drop_all(bind=engine) 
    # Descomentar a linha abaixo para criar as tabelas automaticamente
    
    Base.metadata.create_all(bind=engine)

register_routes(app)