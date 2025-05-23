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
import utilities
# Import the custom handoff tool and LanguageTool integration
from custom_handoff_tools import create_custom_handoff_to_profile
from language_tool import LanguageToolAPI, LanguageToolCorrection

#Initialize the LLM
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# Initialize LanguageTool API
language_tool = LanguageToolAPI()

## Create custom handoff tool
transfer_to_profile = create_custom_handoff_to_profile()

## Schema definitions
class GrammarCorrection(BaseModel):
    """Grammar correction information from LanguageTool"""
    original_text: str = Field(description="The original text with errors", default=None)
    corrected_text: str = Field(description="The corrected text", default=None)
    explanation: Optional[str] = Field(description="Brief explanation of the correction", default=None)
    improvement: str = Field(description="Rewritten user's text in a native-like way", default=None)
    # errors_found: int = Field(description="Number of errors found", default=0)

## Initialize the model and tools
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['corrections']

## Prompts 
MODEL_SYSTEM_MESSAGE = """You are an English language tutor and conversation partner.

Current User Profile:
{user_profile}

Current Corrections:
{corrections}

LanguageTool Analysis (Must be prioritized):
{languagetool_analysis}

MANDATORY ACTIONS - Check in this exact order:

1. **GRAMMAR CHECK**: If LanguageTool found errors OR you detect semantic/flow errors → MUST call UpdateMemory with update_type='corrections'

2. **PERSONAL INFO CHECK**: If user shares name, location, interests, family, job, etc. → MUST call transfer_to_profile and transfer also the user's original message in this pattern: "user's message: `message`"

3. **CONTINUE CONVERSATION**: Only if no personal info → respond naturally using saved profile (may be empty if no information has been collected yet) in a warm, friendly manner as if you're a long-term friend AND if you notice corrections were made to their English, mention them casually and constructively

You MUST prioritize saving grammar corrections. Always call UpdateMemory first when errors are found.

Don't tell the user that you have updated your memory or used LanguageTool. """

TRUSTCALL_INSTRUCTION = """Extract grammar correction information from this interaction.

Include original text, corrected version, explanation in a native way and errors found.(Don't use markdown or formatation style)

System Time: {time}"""

## Node definitions
def corrections_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Main correction agent that checks grammar using LanguageTool and decides on next actions."""
    
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
    
    # Get the user's latest message
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # Check grammar using LanguageTool
    languagetool_result = language_tool.check_text(user_message)
    
    # Format LanguageTool analysis for the prompt
    if languagetool_result.errors:
        languagetool_analysis = f"""
Original text: "{languagetool_result.original_text}"
Corrected text: "{languagetool_result.corrected_text}"
Errors found: {len(languagetool_result.errors)}
Explanation: {languagetool_result.explanation}
"""
    else:
        languagetool_analysis = "No grammar errors found by LanguageTool."
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile, 
        corrections=corrections,
        languagetool_analysis=languagetool_analysis
    )

    # Use parallel tool calls to allow both grammar correction and profile transfer if needed
    response = model.bind_tools(
        [UpdateMemory, transfer_to_profile], 
        parallel_tool_calls=False
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

    spy = utilities.Spy()
    
    ## Create the Trustcall extractors
    trustcall_extractor = create_extractor(
        model,
        tools=[GrammarCorrection],
        tool_choice=tool_name,
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
        print("Correction SAVED\n")
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