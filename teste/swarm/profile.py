import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from trustcall import create_extractor
from typing import Literal, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv(override=True)
import configuration

# Import the custom handoff tool
from custom_handoff_tools import create_custom_handoff_to_correction

#Initialize the LLM
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

## Create custom handoff tool
transfer_to_correction_agent = create_custom_handoff_to_correction()

## Schema definitions
class UserProfile(BaseModel):
    """ Profile of a user """
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    interests: list = Field(description="A list of the user's interests")

## Initialize the model and tools
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['profile']

## Create the Trustcall extractors
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile",
)

## Prompts 
MODEL_SYSTEM_MESSAGE = """You are a friendly English conversation partner focused on getting to know users personally.

You have access to long-term memory about the user's profile and grammar corrections.

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here are recent grammar corrections (for context):
<corrections>
{corrections}
</corrections>

Your responsibilities:
1. **Extract and save personal information** shared by the user (name, location, interests, family, job, etc.)
2. **Continue friendly conversation** using saved profile information
3. **Transfer back to correction agent** if you detect grammar errors

Decision Logic:
- If there IS personal information to extract → Save it to profile memory and respond warmly
- If there are NO grammar errors and NO new personal information → Continue conversation naturally using existing profile
- If there ARE grammar errors → Transfer back to correction agent

Conversation Guidelines:
- Be warm and personable, using the user's name when available
- Reference their interests and personal details naturally in conversation
- Ask follow-up questions about their life, interests, and experiences
- Make them feel heard and remembered
- If you know their interests, occasionally bring up related topics

Remember: Don't tell the user that you updated memory or transferred to another agent."""

TRUSTCALL_INSTRUCTION = """Analyze the following conversation and extract any personal information about the user that should be saved to their profile.

Focus on: name, location, interests, family, job, hobbies, preferences.

System Time: {time}"""

## Node definitions
def profile_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Profile agent that extracts and saves user information."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Get user profile
    namespace = ("memory", user_id)
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    # Get corrections for context
    namespace = ("corrections", user_id)
    memories = store.search(namespace)
    corrections = memories[0].value if memories else None
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile,
        corrections=corrections
    )

    # Use parallel tool calls
    response = model.bind_tools(
        [UpdateMemory, transfer_to_correction_agent], 
        parallel_tool_calls=True
    ).invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": [response]}

def update_memories(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Extract and save user profile information."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

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
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

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
    except Exception as e:
        print(f"Error in profile extraction: {e}")

    # Find the correct tool call ID for profile updates
    tool_calls = state['messages'][-1].tool_calls
    profile_tool_call_id = None
    for tool_call in tool_calls:
        if tool_call['name'] == 'UpdateMemory' and tool_call['args']['update_type'] == "profile":
            profile_tool_call_id = tool_call['id']
            break
    
    if profile_tool_call_id:
        return {"messages": [{"role": "tool", "content": "Profile updated", "tool_call_id": profile_tool_call_id}]}
    else:
        return {"messages": []}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_memories", "tools"]:
    """Route based on the last message in the conversation."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    
    # Check what tools were called
    has_profile_update = False
    has_transfer = False
    
    for tool_call in message.tool_calls:
        if tool_call['name'] == 'UpdateMemory' and tool_call['args']['update_type'] == "profile":
            has_profile_update = True
        elif tool_call['name'] == 'transfer_to_correction_agent':
            has_transfer = True
    
    # If both are called, handle profile update first
    if has_profile_update:
        return "update_memories"
    elif has_transfer:
        return "tools"
    else:
        return END

# Create the ToolNode for handling handoff tools
tools = [transfer_to_correction_agent]
tool_node = ToolNode(tools)

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the profile agent
builder.add_node("profile_agent", profile_agent)
builder.add_node("update_memories", update_memories)
builder.add_node("tools", tool_node)

# Define the flow 
builder.add_edge(START, "profile_agent")
builder.add_conditional_edges("profile_agent", route_message)
builder.add_edge("update_memories", "profile_agent")
builder.add_edge("tools", END)

# Compile the graph
profile_graph = builder.compile(name="Profile")