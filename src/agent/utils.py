from datetime import datetime
from typing import List, Dict, Any, TypedDict, Literal
from uuid import uuid4

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import logging
from ..logging import LogLevels

# Prompts for the agent system
MODEL_SYSTEM_MESSAGE = """You are a helpful, friendly AI assistant who remembers past conversations with users.

You have a memory which keeps track of the user's profile and recent conversations.

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here are recent conversations with this user (may be empty for new users):
<recent_conversations>
{recent_conversations}
</recent_conversations>

Here are your instructions:

1. Be friendly and conversational. Refer to past conversations when appropriate.
2. If you learn personal information about the user, remember it for future conversations.
3. Don't explicitly mention that you've updated your memory unless the user asks.
4. Use information from the user's profile to personalize your responses.
5. If this is a new user, try to learn about their interests and preferences naturally.
"""

PROFILE_EXTRACTION_PROMPT = """Based on the conversation history, please update or create a profile for this user.
Extract information such as:
- The user's name
- Location
- Occupation
- Interests
- Preferences for conversation topics

Only include information that has been explicitly mentioned or can be clearly inferred.
If certain information is not available, leave those fields empty.

System Time: {time}
"""

class UpdateMemory(TypedDict):
    """Decision on what memory type to update."""
    update_type: Literal['profile']


def initialize_model(model_name="gemini-1.5-flash", temperature=0.2):
    """Initialize the LLM model."""
    try:
        return ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    except Exception as e:
        logging.error(f"Failed to initialize LLM model: {str(e)}")
        raise


def extract_profile_from_conversation(model, conversation_history):
    """Extract a user profile from conversation history."""
    from .models import Profile
    
    try:
        # Format the extraction prompt
        prompt = PROFILE_EXTRACTION_PROMPT.format(time=datetime.now().isoformat())
        
        # Combine the prompt with the conversation history
        messages = [
            SystemMessage(content=prompt)
        ] + conversation_history
        
        # Get the model's extraction
        response = model.invoke(messages)
        
        # Parse the response into a Profile object
        profile_data = {}
        
        # Simple parsing - get lines that look like "key: value"
        lines = response.content.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Map to our Profile fields
                if key == 'name':
                    profile_data['name'] = value
                elif key == 'location':
                    profile_data['location'] = value
                elif key == 'occupation' or key == 'job':
                    profile_data['occupation'] = value
                elif key == 'interests':
                    # Split comma-separated interests
                    profile_data['interests'] = [i.strip() for i in value.split(',') if i.strip()]
                elif key == 'preferences':
                    # Split comma-separated preferences
                    profile_data['preferences'] = [p.strip() for p in value.split(',') if p.strip()]
        
        return Profile(**profile_data)
    except Exception as e:
        logging.error(f"Error extracting profile: {str(e)}")
        # Return an empty profile if extraction fails
        return Profile()