import uuid
from datetime import datetime
from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict, List, Dict, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

import configuration
from dotenv import load_dotenv
load_dotenv(override=True)

## Utilities 
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

def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories."""
    
    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        result_parts.append(
            f"New {schema_name} created:\n"
            f"Content: {change['value']}"
        )
    
    if not result_parts:
        return ""
    
    return "\n\n".join(result_parts)

## Schema definitions

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
    topic: str = Field(description="The main topic or subject of conversation")
    timestamp: datetime = Field(description="When this topic was discussed", default_factory=datetime.now)
    user_interest_level: Optional[str] = Field(
        description="How interested the user seemed in this topic (high, medium, low)",
        default=None
    )
    
# Grammar Correction schema
class GrammarCorrection(BaseModel):
    """Record of grammar corrections made for the user"""
    original_text: str = Field(description="The user's original text with errors")
    corrected_text: str = Field(description="The corrected version of the text")
    explanation: str = Field(description="Explanation of the grammar rules and corrections")
    improvement: str = Field(description="Rewritten user's text in a native-like way")
    timestamp: datetime = Field(description="When this correction was made", default_factory=datetime.now)
    
## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'topic', 'grammar']

class WebSearchKnowledge(BaseModel):
    """Knowledge gained from web search to answer user questions"""
    query: str = Field(description="The original search query from the user")
    information: str = Field(description="The factual information gathered from search")
    teaching_notes: Optional[str] = Field(
        description="Notes on how to present this information as an English tutor",
        default=None
    )
    timestamp: datetime = Field(description="When this search was performed", default_factory=datetime.now)


# Initialize the model
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

## Create the Trustcall extractors for each memory type
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
    enable_inserts=True
)

topic_extractor = create_extractor(
    model,
    tools=[ConversationTopic],
    tool_choice="ConversationTopic",
    enable_inserts=True
)

grammar_extractor = create_extractor(
    model,
    tools=[GrammarCorrection],
    tool_choice="GrammarCorrection",
    enable_inserts=True
)

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
<corrections>
{corrections}
</corrections>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. IMPORTANT: ALWAYS CHECK FOR GRAMMAR ERRORS FIRST. This is your primary job as a language tutor!

3. Decide whether any of the your long-term memory should be updated:
- If the user's message contains grammatical errors, ALWAYS call the UpdateMemory tool with type `grammar` to create a new record with the correction details.
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If a new conversation topic was introduced, record it by calling UpdateMemory tool with type `topic`. Don't record the topic if it is already in the list. Only update the user's interests level.

4. Decide what to do next (you will be routed automatically):
- Use grammar correction when errors are detected
- Update memories when new personal information or topics arise
- Don't tell the user that you have updated their profile or your memory
- Continue the conversation naturally, asking follow-up questions where appropriate
- If the user doesn't continue the conversation, use the user's interests to friendly initiate a new topic
   
5. Use an encouraging, supportive tone throughout

6. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made.

"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

## Node definitions

def ai_language_tutor(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

   # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve topics from the store
    namespace = ("topic", user_id)
    memories = store.search(namespace)
    topics = "\n".join(f"{mem.value}" for mem in memories)
    
    # Retrieve grammar corrections from store
    namespace = ("grammar", user_id)
    memories = store.search(namespace)
    corrections = "\n".join(f"{mem.value}" for mem in memories)

    
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, topics=topics, corrections=corrections)

    # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=True).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the profile memory."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    existing_memories = None
    if existing_items:
        existing_memories = [(item.key, "Profile", item.value) for item in existing_items]

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls
    spy = Spy()
    
    # Use the profile extractor with the spy
    local_profile_extractor = profile_extractor.with_listeners(on_end=spy)

    # Invoke the extractor
    result = local_profile_extractor.invoke({
        "messages": updated_messages, 
        "existing": existing_memories
    })

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    
    # Find a tool call ID for this update type
    tool_call_id = None
    for tool_call in state['messages'][-1].tool_calls:
        if tool_call['args']['update_type'] == "user":
            tool_call_id = tool_call['id']
            break
    
    if tool_call_id:
        # Extract changes made by the extractor for reporting
        profile_update_msg = extract_tool_info(spy.called_tools, "Profile")
        if not profile_update_msg:
            profile_update_msg = "Profile updated."
            
        # Return a tool message response
        return {"messages": [{"role": "tool", "content": profile_update_msg, "tool_call_id": tool_call_id}]}
    else:
        return {"messages": []}

def update_topic(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the topic memory."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("topic", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    existing_memories = None
    if existing_items:
        existing_memories = [(item.key, "ConversationTopic", item.value) for item in existing_items]

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls
    spy = Spy()
    
    # Use the topic extractor with the spy
    local_topic_extractor = topic_extractor.with_listeners(on_end=spy)

    # Invoke the extractor
    result = local_topic_extractor.invoke({
        "messages": updated_messages, 
        "existing": existing_memories
    })

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    
    # Find a tool call ID for this update type
    tool_call_id = None
    for tool_call in state['messages'][-1].tool_calls:
        if tool_call['args']['update_type'] == "topic":
            tool_call_id = tool_call['id']
            break
    
    if tool_call_id:
        # Extract changes made by the extractor for reporting
        topic_update_msg = extract_tool_info(spy.called_tools, "ConversationTopic")
        if not topic_update_msg:
            topic_update_msg = "Topic updated."
            
        # Return a tool message response
        return {"messages": [{"role": "tool", "content": topic_update_msg, "tool_call_id": tool_call_id}]}
    else:
        return {"messages": []}

def update_grammar(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the grammar correction memory."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("grammar", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    existing_memories = None
    if existing_items:
        existing_memories = [(item.key, "GrammarCorrection", item.value) for item in existing_items]

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls
    spy = Spy()
    
    # Use the grammar extractor with the spy
    local_grammar_extractor = grammar_extractor.with_listeners(on_end=spy)

    # Invoke the extractor
    result = local_grammar_extractor.invoke({
        "messages": updated_messages, 
        "existing": existing_memories
    })

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    
    # Find a tool call ID for this update type
    tool_call_id = None
    for tool_call in state['messages'][-1].tool_calls:
        if tool_call['args']['update_type'] == "grammar":
            tool_call_id = tool_call['id']
            break
    
    if tool_call_id:
        # Extract changes made by the extractor for reporting
        grammar_update_msg = extract_tool_info(spy.called_tools, "GrammarCorrection")
        if not grammar_update_msg:
            grammar_update_msg = "Grammar correction recorded."
            
        # Return a tool message response
        return {"messages": [{"role": "tool", "content": grammar_update_msg, "tool_call_id": tool_call_id}]}
    else:
        return {"messages": []}


# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> List[Literal[END, "update_topic", "update_grammar", "update_profile"]]: # type: ignore
    """Route the message to the appropriate update function based on tool calls."""
    message = state['messages'][-1]
    
    if len(message.tool_calls) == 0:
        return [END]
    
    # Collect update types needed
    updates_needed = []
    
    # Check each tool call
    for tool_call in message.tool_calls:
        update_type = tool_call['args']['update_type']
        if update_type == "user":
            updates_needed.append("update_profile")
        elif update_type == "topic":
            updates_needed.append("update_topic")
        elif update_type == "grammar":
            updates_needed.append("update_grammar")
    
    # If no updates are needed, end
    if not updates_needed:
        return [END]
    
    return updates_needed

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the memory extraction process flow
builder.add_node(ai_language_tutor)
builder.add_node(update_topic)
builder.add_node(update_profile)
builder.add_node(update_grammar)

# Define the flow 
builder.add_edge(START, "ai_language_tutor")

# Use conditional edges for routing with multiple updates
builder.add_conditional_edges(
    "ai_language_tutor",
    route_message,
    {
        "update_topic": "update_topic",
        "update_profile": "update_profile",
        "update_grammar": "update_grammar",
        END: END
    }
)

# Add returns to ai_language_tutor after each update
builder.add_edge("update_topic", "ai_language_tutor")
builder.add_edge("update_profile", "ai_language_tutor")
builder.add_edge("update_grammar", "ai_language_tutor")

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# Generate Mermaid diagram for visualization
file_name = "graph_mermaid.txt"
with open(file_name, "w") as f:
    f.write(graph.get_graph(xray=1).draw_mermaid())

def main():
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory 
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}
    
    # User introduction
    input_messages = [HumanMessage(content="Hello, My name is Lance. I live in SF with my wife. I have a 1 year old daughter and I like music and sports.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    
    # User mentions music preferences with grammar errors
    input_messages = [HumanMessage(content="I like jazz music and some pop. I listen often when I working. It help me concentrate.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        
    # Check for updated grammar corrections
    user_id = "Lance"
    for memory in across_thread_memory.search(("grammar", user_id)):
        print(memory.value)
    
    # Create a new thread with access to long-term memory
    config = {"configurable": {"thread_id": "2", "user_id": "Lance"}}

    # User mentions sports with grammar errors
    input_messages = [HumanMessage(content="Hello, the Lakers win tonight, I exit my job earlier to watch with my wife")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        
if __name__ == "__main__":
    main()