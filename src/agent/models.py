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

# Search query schema
class WebSearchKnowledge(BaseModel):
    """Knowledge gained from web search to answer user questions"""
    query: str = Field(description="The original search query from the user")
    information: str = Field(description="The factual information gathered from search")
    teaching_notes: Optional[str] = Field(
        description="Notes on how to present this information as an English tutor",
        default=None
    )
    timestamp: datetime = Field(description="When this search was performed", default_factory=datetime.now)
    
# Tools
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['profile', 'topic', 'grammar', 'web_search']
# ----------------------
# API Models 
# ----------------------

# User profile memory model
class UserProfileMemory(BaseModel):
    """
    User profile memory for the agent, includes profile, topics, grammar corrections, and web search memory
    """
    profile: Optional[Dict[str, Any]] = Field(default=None, description="User profile information")
    topics: Optional[List[Dict[str, Any]]] = Field(default=None, description="Conversation topics")
    grammar_corrections: Optional[List[Dict[str, Any]]] = Field(default=None, description="Grammar corrections")
    web_search: Optional[Dict[str, Any]] = Field(default=None, description="Web search memory")


# Agent request model
class AgentRequest(BaseModel):
    """
    Request to the agent for processing a message
    """
    message: str = Field(description="Message to process through the agent")


# Grammar correction response structure
class GrammarCorrectionResponse(BaseModel):
    """
    Grammar correction response structure for the agent
    """
    original_text: str = Field(description="Original text with errors")
    corrected_text: str = Field(description="Corrected version of the text")
    explanation: str = Field(description="Explanation of the grammar correction")


# Agent response model
class AgentResponse(BaseModel):
    """
    Response from the agent after processing a message
    """
    message: str = Field(description="Response message from the agent")
    updated_memory: Optional[UserProfileMemory] = Field(default=None, description="Updated memory after processing")
    grammar_correction: Optional[GrammarCorrectionResponse] = Field(default=None, description="Grammar correction details if applicable")


# Agent memory response model
class AgentMemoryResponse(BaseModel):
    """
    Response containing the agent's memory for a user
    """
    user_id: UUID = Field(description="User ID")
    memory: Optional[UserProfileMemory] = Field(default=None, description="User profile memory")