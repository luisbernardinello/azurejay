from pydantic import BaseModel, Field
from uuid import UUID
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict
import operator

class UserProfileMemory(BaseModel):
    """Profile of a user that's extracted and maintained by the agent"""
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    user_interests: List[str] = Field(description="A list of the user's interests", default_factory=list)

class AgentConfig(BaseModel):
    """Configuration for the agent"""
    user_id: UUID
    conversation_id: UUID
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0

class AgentRequest(BaseModel):
    """Request to process a message with the agent"""
    message: str

class AgentResponse(BaseModel):
    """Response from the agent"""
    message: str
    updated_memory: Optional[UserProfileMemory] = None

class AgentMemoryResponse(BaseModel):
    """Response containing the agent's memory for a user"""
    user_id: UUID
    memory: Optional[UserProfileMemory] = None

# Adicionando à classe EnhancedState
class EnhancedState(TypedDict):
    """Enhanced state that includes context for search results, routing information, and grammar corrections"""
    messages: list
    question: str
    answer: str
    memory: Dict[str, Any]
    context: Annotated[list, operator.add]
    search_needed: bool  # Track if search is needed
    grammar_issues: Optional[Dict[str, Any]] = None  # Track grammar issues if any
    corrected_text: Optional[str] = None  # Store corrected text

# Nova classe para informações de correção gramatical
class GrammarCorrection(BaseModel):
    """Information about grammar corrections made to user input"""
    original_text: str
    corrected_text: str
    issues_found: List[str] = Field(description="List of grammar issues found in the text")
    language_detected: str = Field(description="Detected language of the text")

# Adicionando à classe AgentResponse para incluir informações sobre correções
class AgentResponse(BaseModel):
    """Response from the agent"""
    message: str
    updated_memory: Optional[UserProfileMemory] = None
    grammar_correction: Optional[GrammarCorrection] = None