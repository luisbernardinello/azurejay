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
from custom_handoff_tools import create_custom_handoff_to_profile

#Initialize the LLM
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

## Create custom handoff tool
transfer_to_profile = create_custom_handoff_to_profile()

## Schema definitions
class GrammarCorrection(BaseModel):
    """Grammar correction information"""
    original_text: str = Field(description="The original text with errors")
    corrected_text: str = Field(description="The corrected text")
    explanation: Optional[str] = Field(description="Brief explanation of the correction", default=None)
    timestamp: datetime = Field(description="When the correction was made", default_factory=datetime.now)

## Initialize the model and tools
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['corrections']

## Create the Trustcall extractors
trustcall_extractor = create_extractor(
    model,
    tools=[GrammarCorrection],
    tool_choice="GrammarCorrection",
)

## Prompts 
MODEL_SYSTEM_MESSAGE = """You are an expert English language tutor and friendly conversation partner.

You are designed to help users improve their English through natural conversation and gentle corrections.

You have access to long-term memory about the user's profile and grammar corrections.

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here are the current corrections you've made (may be empty if no corrections have been recorded yet):
<corrections>
{corrections}
</corrections>

Your main responsibilities:
1. **FIRST PRIORITY - Grammar Check**: Always check if the user's message contains grammar, spelling, or punctuation errors
2. **SECOND PRIORITY - Profile Information**: Check if the user shared personal information (name, location, interests, family, job, etc.)

Decision Logic:
- If there ARE grammar errors AND there IS personal information → Save grammar correction first, then transfer to profile agent
- If there ARE grammar errors AND there is NO personal information → Save grammar correction and provide friendly feedback
- If there are NO grammar errors AND there IS personal information → Transfer to profile agent  
- If there are NO grammar errors AND there is NO personal information → Continue conversation naturally

When correcting grammar:
- Be gentle and friendly, especially if you have user profile information
- Ask the user to try saying the corrected version
- Use their name and interests if available from the profile
- Explain the correction in simple terms

When continuing conversation:
- Use information from the user profile to personalize responses
- Ask follow-up questions about their interests
- Be warm and encouraging

Remember: Don't tell the user that you updated memory or transferred to another agent."""

TRUSTCALL_INSTRUCTION = """Analyze the following conversation and extract any grammar corrections that were made.

System Time: {time}"""

## Node definitions
def corrections_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Main correction agent that checks grammar and decides on next actions."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Get user profile
    namespace = ("memory", user_id)
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    # Get corrections history
    namespace = ("corrections", user_id)
    memories = store.search(namespace)
    corrections = memories[0].value if memories else None
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile, 
        corrections=corrections
    )

    # Use parallel tool calls to allow both grammar correction and profile transfer if needed
    response = model.bind_tools(
        [UpdateMemory, transfer_to_profile], 
        parallel_tool_calls=True
    ).invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": [response]}

def update_corrections(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Extract and save grammar corrections."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("corrections", user_id)

    # Retrieve the most recent corrections for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "GrammarCorrection"
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
        print(f"Error in grammar correction extraction: {e}")
    
    # Find the correct tool call ID for corrections
    tool_calls = state['messages'][-1].tool_calls
    correction_tool_call_id = None
    for tool_call in tool_calls:
        if tool_call['name'] == 'UpdateMemory' and tool_call['args']['update_type'] == "corrections":
            correction_tool_call_id = tool_call['id']
            break
    
    if correction_tool_call_id:
        return {"messages": [{"role": "tool", "content": "Grammar correction saved", "tool_call_id": correction_tool_call_id}]}
    else:
        return {"messages": []}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_corrections", "tools"]:
    """Route based on the last message in the conversation."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    
    # Check what tools were called
    has_corrections = False
    has_transfer = False
    
    for tool_call in message.tool_calls:
        if tool_call['name'] == 'UpdateMemory' and tool_call['args']['update_type'] == "corrections":
            has_corrections = True
        elif tool_call['name'] == 'transfer_to_profile':
            has_transfer = True
    
    # If both are called, handle corrections first
    if has_corrections:
        return "update_corrections"
    elif has_transfer:
        return "tools"
    else:
        return END

# Create the ToolNode for handling handoff tools
tools = [transfer_to_profile]
tool_node = ToolNode(tools)

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the correction agent
builder.add_node("corrections_agent", corrections_agent)
builder.add_node("update_corrections", update_corrections)
builder.add_node("tools", tool_node)

# Define the flow 
builder.add_edge(START, "corrections_agent")
builder.add_conditional_edges("corrections_agent", route_message)
builder.add_edge("update_corrections", "corrections_agent")
builder.add_edge("tools", END)

# Compile the graph
correction_graph = builder.compile(name="Correction")