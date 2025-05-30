import operator
import uuid
from datetime import datetime
from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Annotated, Literal, Optional, TypedDict, Dict, Any, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_groq import ChatGroq

import configuration
from dotenv import load_dotenv
load_dotenv(override=True)

## Utilities 
# Inspect the tool calls for Trustcall
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

## Schema definitions

# User profile schema
class UserProfile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    interests: list[str] = Field(
        description="Topics the user is interested in discussing in English", 
        default_factory=list,
        max_items=10
    )
    
# Grammar Correction schema
class GrammarCorrection(BaseModel):
    """Record of grammar corrections made for the user"""
    original_text: str = Field(description="The user's original text with errors")
    corrected_text: str = Field(description="The corrected version of the text")
    explanation: str = Field(description="Explanation of the grammar rules and corrections")
    improvement: str = Field(description="Rewritten user's text in a native-like way")

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'grammar', 'both']

# Initialize the model
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

## Prompts 

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """You are an expert English language tutor and friendly conversation partner.

You have access to context from previous interactions, including:
- Corrections that may have been made to their original input
- Research information that may be relevant to their question
- Your long-term memory about the user

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here are recent Grammar Corrections you've provided (may be empty if no corrections were needed):
<grammar>
{grammar}
</grammar>

Context from previous processing:
<context>
{context}
</context>

Your instructions for reasoning about the user's messages:

1. **MEMORY UPDATES** - Check what needs to be saved:
   - If the user provided personal information (name, location, interests): Call UpdateMemory with 'user'
   - If grammar corrections were made (check context for corrections): Call UpdateMemory with 'grammar'  
   - If both personal info AND corrections are present: Call UpdateMemory with 'both'

2. **RESPONSE** - After handling memory updates, provide a warm, friendly response that:
   - Acknowledges the user's input naturally
   - If corrections were made, mention them casually and constructively
   - If research information is available, present it helpfully
   - Continues the conversation naturally with follow-up questions where appropriate
   - Uses the user's interests to initiate new topics if conversation stalls
   - Maintains an encouraging tone about their language learning journey

Important guidelines:
- Don't tell the user that you have updated their profile or memory
- Be encouraging and supportive
- Focus on natural conversation flow
- Help them learn English through engaging dialogue
"""

# Trustcall instruction
PROFILE_TRUSTCALL_INSTRUCTION = """Extract personal information from this conversation: name, location, and interests.

If profile already exists, only update the interests list and don't include the same interest twice.

System Time: {time}"""

CORRECTION_TRUSTCALL_INSTRUCTION = """Extract grammar correction information from this interaction.

Include:
- Original text with errors
- Corrected version  
- Clear explanation of grammar rules
- Improved native-like rewrite

Don't use markdown or special formatting.

System Time: {time}"""

## Task definitions

@task
def get_user_memories(user_id: str, *, store: BaseStore) -> dict:
    """Retrieve user profile and grammar corrections from the store."""
    
    # Retrieve profile memory from the store
    namespace = ("memory", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = "No profile information available yet."

    # Retrieve grammar corrections from store
    namespace = ("corrections", user_id)
    memories = store.search(namespace)
    grammar = "\n".join(f"{mem.value}" for mem in memories) if memories else "No previous corrections."
    
    return {
        "user_profile": user_profile,
        "grammar": grammar
    }

@task
def generate_ai_response(messages: list, memories: dict) -> dict:
    """Generate AI tutor response with memory context."""
    
    # Build context from previous messages in the conversation
    context_parts = []
    for msg in messages:
        if hasattr(msg, 'name'):
            if msg.name == "correction" and msg.content != "CORRECT":
                context_parts.append(f"Grammar correction made: {msg.content}")
            elif msg.name == "researcher":
                context_parts.append(f"Research information: {msg.content}")
            elif msg.name == "enhancer":
                context_parts.append(f"Enhanced query: {msg.content}")
    
    context = "\n".join(context_parts) if context_parts else "No additional context from previous processing."
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=memories["user_profile"], 
        grammar=memories["grammar"],
        context=context
    )

    # Get only the original user message for response
    user_messages = [msg for msg in messages if not hasattr(msg, 'name') or msg.name not in ["correction", "researcher", "enhancer", "supervisor", "validator"]]
    
    # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)] + user_messages)

    return {
        "response": response,
        "has_tool_calls": bool(response.tool_calls),
        "update_type": response.tool_calls[0]['args'].get('update_type') if response.tool_calls else None
    }

@task
def update_user_profile(messages: list, user_id: str, *, store: BaseStore) -> str:
    """Extract and save user profile information."""
    
    # Define the namespace for the memories
    namespace = ("memory", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "UserProfile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = PROFILE_TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + messages[:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    ## Create the Trustcall extractors
    trustcall_extractor = create_extractor(
        model,
        tools=[UserProfile],
        tool_choice="UserProfile",
    ).with_listeners(on_end=spy)
    
    try:
        # Invoke the extractor
        result = trustcall_extractor.invoke({"messages": updated_messages, 
                                             "existing": existing_memories})

        # Save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace,
                      rmeta.get("json_doc_id", str(uuid.uuid4())),
                      r.model_dump(mode="json"),
                )
        
        print("--- Profile information updated ---")
        return "Profile information updated successfully"
        
    except Exception as e:
        print(f"Error in profile extraction: {e}")
        return f"Error updating profile: {e}"

@task
def update_grammar_corrections(messages: list, user_id: str, *, store: BaseStore) -> str:
    """Extract and save grammar correction information."""
    
    # Define the namespace for the memories
    namespace = ("corrections", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "GrammarCorrection"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = CORRECTION_TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + messages[:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    ## Create the Trustcall extractors
    trustcall_extractor = create_extractor(
        model,
        tools=[GrammarCorrection],
        tool_choice="GrammarCorrection",
        enable_inserts=True
    ).with_listeners(on_end=spy)

    try:
        # Invoke the extractor
        result = trustcall_extractor.invoke({"messages": updated_messages, 
                                             "existing": existing_memories})

        # Save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace,
                      rmeta.get("json_doc_id", str(uuid.uuid4())),
                      r.model_dump(mode="json"),
                )
            
        print("--- Grammar correction information updated ---")
        
        # Extract the changes made by Trustcall
        grammar_correction_update_msg = extract_tool_info(spy.called_tools, tool_name)
        return grammar_correction_update_msg or "Grammar correction updated successfully"
        
    except Exception as e:
        print(f"Error in grammar correction extraction: {e}")
        return f"Error updating grammar corrections: {e}"

## Entrypoint definition

@entrypoint(checkpointer=MemorySaver())
def responder_workflow(input_data: dict, *, config: RunnableConfig, store: BaseStore) -> dict:
    """Main responder workflow that handles user interactions with memory management."""
    
    # Extract configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Extract messages from input
    messages = input_data.get("messages", [])
    
    # Step 1: Get user memories
    memories_future = get_user_memories(user_id, store=store)
    memories = memories_future.result()
    
    # Step 2: Generate AI response
    ai_response_future = generate_ai_response(messages, memories)
    ai_response_data = ai_response_future.result()
    
    # Step 3: Handle memory updates if needed
    update_messages = []
    
    if ai_response_data["has_tool_calls"]:
        update_type = ai_response_data["update_type"]
        
        if update_type == "user":
            profile_future = update_user_profile(messages, user_id, store=store)
            profile_result = profile_future.result()
            update_messages.append({
                "role": "tool", 
                "content": profile_result, 
                "tool_call_id": ai_response_data["response"].tool_calls[0]['id']
            })
            
        elif update_type == "grammar":
            grammar_future = update_grammar_corrections(messages, user_id, store=store)
            grammar_result = grammar_future.result()
            update_messages.append({
                "role": "tool", 
                "content": grammar_result, 
                "tool_call_id": ai_response_data["response"].tool_calls[0]['id']
            })
            
        elif update_type == "both":
            # Update both profile and grammar in parallel
            profile_future = update_user_profile(messages, user_id, store=store)
            grammar_future = update_grammar_corrections(messages, user_id, store=store)
            
            profile_result = profile_future.result()
            grammar_result = grammar_future.result()
            
            update_messages.append({
                "role": "tool", 
                "content": "Both profile and grammar corrections updated successfully", 
                "tool_call_id": ai_response_data["response"].tool_calls[0]['id']
            })
        
        # If we have tool calls, we need to generate a final response
        if update_messages:
            # Create new messages list with the tool response
            final_messages = messages + [ai_response_data["response"]] + update_messages
            
            # Generate final response after memory updates
            final_response_future = generate_ai_response(final_messages, memories)
            final_response_data = final_response_future.result()
            
            return {
                "messages": final_messages + [final_response_data["response"]]
            }
    
    # If no tool calls or updates needed, return the original response
    return {
        "messages": messages + [ai_response_data["response"]]
    }

# Create function to use the entrypoint (compatibility with existing code)
def create_responder_subgraph():
    """
    Returns the compiled responder workflow entrypoint.
    
    The @entrypoint decorator has already transformed 'responder_workflow'
    into a runnable Pregel object that has an .invoke() method.
    We just need to return it directly.
    
    The 'store' argument is kept for compatibility with supervisor.py, 
    but LangGraph will inject the store automatically into the entrypoint
    since it's compiled in the main graph.
    """
    return responder_workflow