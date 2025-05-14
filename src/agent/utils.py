import json
import logging
import redis
from typing import Dict, List, Tuple, Any, Optional
from langchain_groq import ChatGroq
from trustcall import create_extractor

from src.agent.models import Profile

class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.
    
    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ConversationTopic", "Profile")
    """
    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts)

# Initialize the model
def initialize_model():
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    return model

# Create the Trustcall extractors for updating the user profile
def get_profile_extractor():
    model = initialize_model()
    profile_extractor = create_extractor(
        model,
        tools=[{
            "type": "function",
            "function": {
                "name": "Profile",
                "description": "Tool to add or update information about the user's profile",
                "parameters": Profile.model_json_schema()
            }
        }],
        tool_choice="Profile",
        enable_inserts=True
    )
    
    return profile_extractor

## Prompts 

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """You are a expert English language tutor. 

You are designed to be a friend to a user, helping them to improve their English through natural conversation, gentle corrections, and helpful explanations.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. Conversation topics you've discussed with them
3. Grammar corrections you've provided them

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here are recent Conversation Topics (may be empty if this is a new user):
<topics>
{topics}
</topics>

Here are recent Grammar Corrections you've provided (may be empty if no corrections were needed):
<grammar>
{grammar}
</grammar>

Here are the web search knowledge (may be empty if no questions were asked):
<web_search>
{web_search}
</web_search>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If the user's message contains grammatical errors, call the UpdateMemory tool with type `grammar` to record the correction details.
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If a new conversation topic was introduced, record it by calling UpdateMemory tool with type `topic`. Don't record the topic if it is already in the list. Only update the user's interests level.
- If the user asks a question that requires factual information, call the UpdateMemory tool with type `web_search` to search for an answer.

3. Decide what to do next (you will be routed automatically):
- Use grammar correction when errors are detected
- Use web search by calling UpdateMemory tool with type `web_search` when questions need external information.
- After performing a web search, check whether the topic is already in the topics list and add it if it's not.
- Update memories when new personal information or topics arise.
- Otherwise, simply continue the conversation 

4. When answering factual questions:
- Use the web search information to provide accurate answers
- Present the information in a way that helps the user learn English
- Point out useful vocabulary from the subject matter
- Maintain your friendly, supportive tutor tone
- Consider suggesting follow-up questions that would help the user practice discussing the topic

5. After any memory updates or searches, or if no tool call was made, respond naturally to the user:
- If you made grammar corrections, politely point them out with explanations
- If you answered a question, provide clear, helpful information
- Continue the conversation naturally, asking follow-up questions where appropriate
- If the user doesn't continue the conversation, use the user's interests to friendly initiate a new topic
- Use an encouraging, supportive tone throughout
- Don't tell the user that you have updated their profile or your memory

"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""