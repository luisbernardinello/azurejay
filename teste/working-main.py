import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, TypedDict, List, Dict, Any, Sequence

from trustcall import create_extractor

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import add_messages  # Helper function to add messages to the state
from langchain_groq import ChatGroq


import configuration
from dotenv import load_dotenv
load_dotenv(override=True)

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
    
# Web Search Knowledge schema
class WebSearchKnowledge(BaseModel):
    """Knowledge gained from web search to answer user questions"""
    query: str = Field(description="The original search query from the user")
    information: str = Field(description="The factual information gathered from search")
    teaching_notes: Optional[str] = Field(
        description="Notes on how to present this information as an English tutor",
        default=None
    )
    timestamp: datetime = Field(description="When this search was performed", default_factory=datetime.now)

## ReAct State
class TutorState(TypedDict):
    """The state of the language tutor agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Using the add_messages helper
    memory_updates: List[Dict[str, Any]]  # Track memory updates
    step_count: int  # Track number of steps taken

## Memory Tool Definitions
class ProfileTool(BaseModel):
    """Tool for updating the user's profile"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: Optional[List[str]] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default=None
    )
    english_level: Optional[str] = Field(
        description="The user's English proficiency level (e.g., beginner, intermediate, advanced)",
        default=None
    )
    interests: Optional[List[str]] = Field(
        description="Topics the user is interested in discussing in English", 
        default=None
    )

class TopicTool(BaseModel):
    """Tool for updating conversation topics"""
    topic: str = Field(description="The main topic of conversation")
    user_interest_level: str = Field(
        description="How interested the user seemed in this topic (high, medium, low)"
    )

class GrammarTool(BaseModel):
    """Tool for creating grammar corrections"""
    original_text: str = Field(description="The user's original text with errors")
    corrected_text: str = Field(description="The corrected version of the text")
    explanation: str = Field(description="Explanation of the grammar rules and corrections")
    improvement: str = Field(description="Rewritten user's text in a native-like way")

## Initialize model and tools

# Initialize the model
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

## Prompts
MODEL_SYSTEM_MESSAGE = """You are an expert English language tutor. 

You are designed to be a friend to a user, helping them to improve their English through natural conversation.

Here are your instructions for reasoning about the user's messages:

YOUR PRIMARY RESPONSIBILITY is to detect and correct grammar errors in the user's English. Then engage in friendly conversation.

1. WHENEVER the user writes a message with ANY grammar errors:
- You MUST use the grammar_tool to record the error
- You MUST explain the correction clearly in your response
- You MUST offer an improved, more natural-sounding version

2. GATHER USER INFORMATION STRATEGICALLY
   - When the user shares personal details, use the profile_tool to update their profile
   - Record conversation topics with the topic_tool when discussing new subjects
   - Don't explicitly tell the user you're updating your records

3. BE WARM AND CONVERSATIONAL
   - Use an encouraging, supportive tone throughout
   - Ask natural follow-up questions based on what you've learned about the user
   - If the conversation stalls, bring up topics based on their interests

You have access to three tools:
1. profile_tool: Update or create user profile information
2. topic_tool: Record conversation topics and user interest levels
3. grammar_tool: Record grammar corrections with explanations

Use these tools appropriately to provide the best language tutoring experience.


"""

## Tool Definitions
from langchain_core.tools import Tool, tool

@tool
def profile_tool(name: Optional[str] = None, location: Optional[str] = None, 
                job: Optional[str] = None, connections: Optional[List[str]] = None,
                english_level: Optional[str] = None, interests: Optional[List[str]] = None):
    """Update the user's profile with any provided information.
    
    Args:
        name: The user's name
        location: The user's location 
        job: The user's job
        connections: Personal connections like family, friends, coworkers
        english_level: English proficiency (beginner, intermediate, advanced)
        interests: Topics the user likes discussing in English
    """
    profile_data = {
        "name": name,
        "location": location,
        "job": job,
        "connections": connections,
        "english_level": english_level,
        "interests": interests
    }
    # Filter out None values
    profile_data = {k: v for k, v in profile_data.items() if v is not None}
    return f"Updated user profile with: {profile_data}"

@tool
def topic_tool(topic: str, user_interest_level: str):
    """Record a conversation topic and the user's interest level in it.
    
    Args:
        topic: The main topic or subject of conversation
        user_interest_level: How interested the user seemed (high, medium, low)
    """
    return f"Recorded topic '{topic}' with interest level: {user_interest_level}"

@tool
def grammar_tool(original_text: str, corrected_text: str, explanation: str, improvement: str):
    """Record a grammar correction with explanation.
    
    Args:
        original_text: The user's original text with errors
        corrected_text: The corrected version of the text
        explanation: Explanation of the grammar rules and corrections
        improvement: Rewritten user's text in a native-like way
    """
    return f"Recorded grammar correction:\nOriginal: {original_text}\nCorrected: {corrected_text}"

tools = [profile_tool, topic_tool, grammar_tool]

# Bind tools to the model
model_with_tools = model.bind_tools(tools)

## Node functions for the ReAct pattern

def call_model(state: TutorState, config: RunnableConfig, store: BaseStore):
    """Node to call the language model with tools."""
    
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
    
    # Create a system message with memory context
    system_msg = f"""{MODEL_SYSTEM_MESSAGE}

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

"""

    # Increase step count
    step_count = state.get("step_count", 0) + 1
    
    # Call the model with the system message and conversation history
    response = model_with_tools.invoke([SystemMessage(content=system_msg)] + state["messages"])
    
    return {"messages": [response], "step_count": step_count}

def update_profile(state: TutorState, config: RunnableConfig, store: BaseStore):
    """Node to update the user profile."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Get the last message with tool calls
    last_message = state["messages"][-1]
    tool_outputs = []
    
    # Process only profile tool calls
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "profile_tool":
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            # Update profile in store
            namespace = ("profile", user_id)
            store.put(namespace, str(uuid.uuid4()), tool_args)
            result = f"Updated user profile information"
            
            # Create tool message
            tool_outputs.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_id,
                    name="profile_tool"
                )
            )
    
    # Return the tool outputs to be added to messages
    return {"messages": tool_outputs}

def update_topic(state: TutorState, config: RunnableConfig, store: BaseStore):
    """Node to update conversation topics."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Get the last message with tool calls
    last_message = state["messages"][-1]
    tool_outputs = []
    
    # Process only topic tool calls
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "topic_tool":
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            # Store topic in store
            namespace = ("topic", user_id)
            topic_data = {
                "topic": tool_args["topic"],
                "user_interest_level": tool_args["user_interest_level"],
                "timestamp": datetime.now().isoformat()
            }
            store.put(namespace, str(uuid.uuid4()), topic_data)
            result = f"Recorded topic '{tool_args['topic']}'"
            
            # Create tool message
            tool_outputs.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_id,
                    name="topic_tool"
                )
            )
    
    # Return the tool outputs to be added to messages
    return {"messages": tool_outputs}

def update_grammar(state: TutorState, config: RunnableConfig, store: BaseStore):
    """Node to update grammar corrections."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Get the last message with tool calls
    last_message = state["messages"][-1]
    tool_outputs = []
    
    # Process only grammar tool calls
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "grammar_tool":
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            # Store grammar correction in store
            namespace = ("grammar", user_id)
            grammar_data = {
                "original_text": tool_args["original_text"],
                "corrected_text": tool_args["corrected_text"],
                "explanation": tool_args["explanation"],
                "improvement": tool_args["improvement"],
                "timestamp": datetime.now().isoformat()
            }
            store.put(namespace, str(uuid.uuid4()), grammar_data)
            result = f"Recorded grammar correction"
            
            # Create tool message
            tool_outputs.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_id,
                    name="grammar_tool"
                )
            )
    
    # Return the tool outputs to be added to messages
    return {"messages": tool_outputs}

## Edge conditions

def route_tools(state: TutorState):
    """Route to the appropriate tool nodes based on tool calls in the last message."""
    messages = state["messages"]
    
    # If there are no messages or no tool calls, end the flow
    if not messages or not hasattr(messages[-1], "tool_calls") or not messages[-1].tool_calls:
        return [END]
    
    # Collect the tool names that were called
    routes = []
    tool_names = set(tool_call["name"] for tool_call in messages[-1].tool_calls)
    
    # Route to the appropriate tool nodes
    if "profile_tool" in tool_names:
        routes.append("profile")
    if "topic_tool" in tool_names:
        routes.append("topic")
    if "grammar_tool" in tool_names:
        routes.append("grammar")
    
    # If no valid tools were called, end the flow
    if not routes:
        return [END]
    
    return routes

## Create the graph

# Create the graph with our state
workflow = StateGraph(TutorState, config_schema=configuration.Configuration)

# Add the nodes
workflow.add_node("tutor", call_model)
workflow.add_node("profile", update_profile)
workflow.add_node("topic", update_topic)
workflow.add_node("grammar", update_grammar)

# Set the entry point
workflow.set_entry_point("tutor")

# Add conditional edges with multiple possible destinations
workflow.add_conditional_edges(
    # After the tutor node
    "tutor",
    # The function that routes to tools
    route_tools,
    # Mapping of where to go next
    # {
    #     "profile": "profile",
    #     "topic": "topic",
    #     "grammar": "grammar",
    #     END: END,
    # },
)

# Add edges from each tool back to tutor
workflow.add_edge("profile", "tutor")
workflow.add_edge("topic", "tutor")
workflow.add_edge("grammar", "tutor")

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer and store
graph = workflow.compile(
    checkpointer=within_thread_memory, 
    store=across_thread_memory
)

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
    for chunk in graph.stream({"messages": input_messages, "step_count": 0, "memory_updates": []}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    
    # User mentions music preferences with grammar errors
    input_messages = [HumanMessage(content="I like jazz music and some pop. I listen often when I working. It help me concentrate.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages, "step_count": 0, "memory_updates": []}, config, stream_mode="values"):
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
    for chunk in graph.stream({"messages": input_messages, "step_count": 0, "memory_updates": []}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        
if __name__ == "__main__":
    main()