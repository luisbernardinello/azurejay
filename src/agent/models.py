from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from typing_extensions import TypedDict
from uuid import UUID

# ----------------------
# LangGraph Agent Models
# ----------------------

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: List[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    english_level: Optional[str] = Field(
        description="The user's English proficiency level (e.g., beginner, intermediate, advanced)",
        default=None
    )
    interests: List[str] = Field(
        description="Topics the user is interested in discussing in English", 
        default_factory=list,
        max_items=10
    )
    
# Conversation Topic schema
class ConversationTopic(BaseModel):
    """Record of topics discussed with the English learner"""
    topic: str = Field(description="The main topic or subject of conversation", default=None)
    timestamp: datetime = Field(description="When this topic was discussed", default_factory=datetime.now)
    user_interest_level: Optional[str] = Field(
        description="How interested the user seemed in this topic (high, medium, low)",
        default=None
    )
    
# Grammar Correction schema
class GrammarCorrection(BaseModel):
    """Record of grammar corrections made for the user"""
    original_text: str = Field(description="The user's original text with errors", default=None)
    corrected_text: str = Field(description="The corrected version of the text", default=None)
    explanation: str = Field(description="Explanation of the grammar rules and corrections", default=None)
    improvement: str = Field(description="Rewritten user's text in a native-like way", default=None)
    timestamp: datetime = Field(description="When this correction was made", default_factory=datetime.now)
    
# Tools
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['profile', 'topic', 'grammar', 'web_search']

# Search query tool
class WebSearchKnowledge(BaseModel):
    """Knowledge gained from web search to answer user questions"""
    query: str = Field(description="The original search query from the user")
    information: str = Field(description="The factual information gathered from search")
    teaching_notes: Optional[str] = Field(
        description="Notes on how to present this information as an English tutor",
        default=None
    )
    timestamp: datetime = Field(description="When this search was performed", default_factory=datetime.now)

# Enhanced State for the agent
class EnhancedState(BaseModel):
    messages: List[Any]
    question: str
    answer: str
    memory: Dict[str, Any] = Field(default_factory=dict)
    context: List[str] = Field(default_factory=list)
    search_needed: bool = False
    grammar_issues: Optional[Dict[str, Any]] = None
    corrected_text: Optional[str] = None

# ----------------------
# API Models 
# ----------------------

# Agent Request model
class AgentRequest(BaseModel):
    """
    Request model for the agent API
    """
    message: str

# User Profile Memory model
class UserProfileMemory(BaseModel):
    """
    Model for user profile memory
    """
    profile: Optional[Dict[str, Any]] = None
    topics: List[Dict[str, Any]] = Field(default_factory=list)
    grammar_corrections: List[Dict[str, Any]] = Field(default_factory=list)
    web_search: Optional[Dict[str, Any]] = None

# Agent Response model
class AgentResponse(BaseModel):
    """
    Response model for the agent API
    """
    message: str
    updated_memory: Optional[UserProfileMemory] = None
    grammar_correction: Optional[Dict[str, Any]] = None

# Agent Memory Response model
class AgentMemoryResponse(BaseModel):
    """
    Response model for the agent memory API
    """
    user_id: UUID
    memory: Optional[UserProfileMemory] = None