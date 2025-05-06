from fastapi import FastAPI
from src.auth.controller import router as auth_router
from src.users.controller import router as users_router
from src.conversations.controller import router as conversations_router
from src.agent.controller import router as agent_router

def register_routes(app: FastAPI):
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(conversations_router)
    app.include_router(agent_router)