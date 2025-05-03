from fastapi import FastAPI
from .database.core import Base, engine, verify_database_connections
from .entities.user import User  # Import models to register them
from .api import register_routes
from .logging import configure_logging, LogLevels


configure_logging(LogLevels.info)

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    # Verificar todas as conexões de banco de dados na inicialização
    verify_database_connections()
    
    # Descomentar a linha abaixo para criar as tabelas automaticamente
    Base.metadata.create_all(bind=engine)

register_routes(app)