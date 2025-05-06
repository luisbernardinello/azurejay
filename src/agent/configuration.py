from pydantic import BaseModel, Field
from uuid import UUID
from langchain_core.runnables.config import RunnableConfig

class Configuration(BaseModel):
    """Configuration for agent runs"""
    user_id: str
    conversation_id: str


    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "Configuration":
        """
        Extract the configurable part from a RunnableConfig
        """
        return cls(**config.get("configurable", {}))