import logging
import os
from typing import Annotated

import redis
from dotenv import load_dotenv
from fastapi import Depends
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()

# =========================
# PostgreSQL Configuration
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
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

# =========================
# Redis & LangGraph Configuration
# =========================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- 1. Standard Redis Client (for general purpose use) ---
try:
    redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
    redis_client_instance = redis.Redis(connection_pool=redis_pool)
    redis_client_instance.ping()  # Verify connection
    logging.info("Standard Redis client connection successful.")
except Exception as e:
    logging.error(f"Failed to connect standard Redis client: {e}")
    redis_client_instance = None

def get_redis_client():
    if redis_client_instance is None:
        raise ConnectionError("Redis connection was not established at startup.")
    return redis_client_instance

RedisDep = Annotated[redis.Redis, Depends(get_redis_client)]


# --- 2. LangGraph Checkpointer (short-term conversation memory) ---
# Following the documentation to initialize at application startup.
try:
    with RedisSaver.from_conn_string(REDIS_URL) as cp:
        # The .setup() method is REQUIRED to create the necessary Redis structures.
        cp.setup()
        checkpointer_instance = cp
        logging.info("LangGraph Checkpointer (RedisSaver) initialized and set up successfully.")
except Exception as e:
    logging.critical(f"Critical failure initializing LangGraph Checkpointer: {e}")
    checkpointer_instance = None

def get_checkpointer() -> RedisSaver:
    """Dependency provider for the LangGraph checkpointer."""
    if checkpointer_instance is None:
        raise ConnectionError("LangGraph Checkpointer was not initialized correctly.")
    return checkpointer_instance


CheckpointerDep = Annotated[RedisSaver, Depends(get_checkpointer)]


# --- 3. LangGraph Store (long-term user memory) ---
# The pattern is the same for the Store.
try:
    with RedisStore.from_conn_string(REDIS_URL) as s:
        # .setup() is also required here.
        s.setup()
        store_instance = s
        logging.info("LangGraph Store (RedisStore) initialized and set up successfully.")
except Exception as e:
    logging.critical(f"Critical failure initializing LangGraph Store: {e}")
    store_instance = None

def get_store() -> BaseStore:
    """Dependency provider for the LangGraph store."""
    if store_instance is None:
        raise ConnectionError("LangGraph Store was not initialized correctly.")
    return store_instance

StoreDep = Annotated[BaseStore, Depends(get_store)]


# =========================
# Database Verification Function (called in main.py on startup)
# =========================
def verify_database_connections():
    """
    Confirms the status of database connections.
    The primary connection logic has already run at the module level.
    This function serves as a startup confirmation.
    """
    logging.info("Verifying database connection statuses...")
    if engine:
        logging.info("PostgreSQL Engine: OK")
    if redis_client_instance:
        logging.info("Redis Client: OK")
    if checkpointer_instance:
        logging.info("LangGraph Checkpointer: OK")
    if store_instance:
        logging.info("LangGraph Store: OK")