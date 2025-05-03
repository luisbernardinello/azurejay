from typing import Annotated
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import os
import redis
import logging
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# PostgreSQL Configuration
# -------------------------

""" You can add a DATABASE_URL environment variable to your .env file """
DATABASE_URL = os.getenv("DATABASE_URL")

""" Or hard code SQLite here """
if DATABASE_URL is None:
    DATABASE_URL = "sqlite:///./todosapp.db"
    # Uncomment below to use SQLite in-memory database
    # DATABASE_URL = "sqlite:///:memory:"

# """ Or hard code PostgreSQL here """
# DATABASE_URL="postgresql://postgres:postgres@db:5432/cleanfastapi"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
DbSession = Annotated[Session, Depends(get_db)]

# -------------------------
# Redis Configuration
# -------------------------

# Obter a URL do Redis do ambiente ou usar o padrão local
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Criar o pool de conexões Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

def get_redis():
    """
    Provedor de dependência para obter uma conexão Redis
    """
    try:
        # Verificar conexão
        redis_client.ping()
        yield redis_client
    except redis.RedisError as e:
        logging.error(f"Redis connection error: {str(e)}")
        raise
    finally:
        # Não há necessidade de fechar explicitamente, o pool gerencia isso
        pass

RedisClient = Annotated[redis.Redis, Depends(get_redis)]

# -------------------------
# Database Initialization and Verification
# -------------------------

def verify_database_connections():
    """
    Verify all database connections
    """
    # Check PostgreSQL
    try:
        with engine.connect() as conn:
            logging.info("PostgreSQL connection successful")
    except Exception as e:
        logging.error(f"Failed to connect to PostgreSQL: {str(e)}")
    
    # Check Redis
    try:
        redis_client.ping()
        logging.info("Redis connection successful")
    except Exception as e:
        logging.error(f"Failed to connect to Redis: {str(e)}")