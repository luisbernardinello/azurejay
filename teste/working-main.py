import operator
import uuid
from datetime import datetime
from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Annotated, Literal, Optional, TypedDict, Dict, Any, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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
load_dotenv()
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
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    english_level: Optional[str] = Field(
        description="The user's English proficiency level (e.g., beginner, intermediate, advanced)",
        default=None
    )
    interests: list[str] = Field(
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
    
## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'topic', 'grammar', 'web_search']

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


# Initialize the model
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3)

## Create the Trustcall extractors for updating the user profile
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
- If a new conversation topic was introduced, record it by calling UpdateMemory tool with type `topic`
- If the user asks a question that requires factual information, call the UpdateMemory tool with type `web_search` to search for an answer

3. Decide what to do next (you will be routed automatically):
- Use grammar correction when errors are detected
- Use web search by calling UpdateMemory tool with type `web_search` when questions need external information
- Update memories when new personal information or topics arise
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
    grammar = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve web search knowledge from store
    namespace = ("web_search", user_id)
    memories = store.search(namespace)
    if memories:
        web_search = memories[0].value.get("memory", "")
    else:
        web_search = ""
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile, 
        topics=topics, 
        grammar=grammar, 
        web_search=web_search
    )

    # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Invoke the extractor
    result = profile_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    
    # Get the tool call ID from the message
    tool_calls = state['messages'][-1].tool_calls
    tool_call_id = tool_calls[0]['id'] if tool_calls and len(tool_calls) > 0 else "unknown"
    
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "Updated profile information", "tool_call_id": tool_call_id}]}

def update_topic(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("topic", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Topic"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    # Create the Trustcall extractor for updating the Topic list 
    topic_extractor = create_extractor(
        model,
        tools=[{
            "type": "function",
            "function": {
                "name": "Topic",
                "description": "Tool to add or update topics in the user's Topic list",
                "parameters": ConversationTopic.model_json_schema()
            }
        }],
        tool_choice="Topic",
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = topic_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
        
    # Get the tool call ID from the message
    tool_calls = state['messages'][-1].tool_calls
    tool_call_id = tool_calls[0]['id'] if tool_calls and len(tool_calls) > 0 else "unknown"
        
    # Extract the changes made by Trustcall and add the the ToolMessage returned to ai_language_tutor
    topic_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": topic_update_msg or "Topic updated", "tool_call_id": tool_call_id}]}

def update_grammar(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("grammar", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Grammar"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    # Create the Trustcall extractor for updating the Grammar list 
    grammar_extractor = create_extractor(
        model,
        tools=[{
            "type": "function",
            "function": {
                "name": "Grammar",
                "description": "Tool to add or update grammar errors in the user's GrammarCorrection list",
                "parameters": GrammarCorrection.model_json_schema()
            }
        }],
        tool_choice="Grammar",
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = grammar_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
        
    # Get the tool call ID from the message
    tool_calls = state['messages'][-1].tool_calls
    tool_call_id = tool_calls[0]['id'] if tool_calls and len(tool_calls) > 0 else "unknown"
    
    # Extract the changes made by Trustcall and add the the ToolMessage returned to ai_language_tutor
    grammar_correction_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": grammar_correction_update_msg or "Grammar updated", "tool_call_id": tool_call_id}]}

def web_search_api(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Search the web for information to answer user questions while maintaining the English tutor role."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    namespace = ("web_search", user_id)

    # Get question from user's last message
    user_messages = [msg for msg in state['messages'] if hasattr(msg, 'type') and msg.type == 'human']
    question = user_messages[-1].content if user_messages else "Recent factual information"
    
    try:
        # Tavily search
        tavily_search = TavilySearchResults(max_results=1)
        tavily_search_docs = tavily_search.invoke(question)
        
        # Wikipedia search
        wikipedia_search_docs = WikipediaLoader(query=question, load_max_docs=1).load()
        
        # Format Tavily results
        tavily_formatted = []
        for doc in tavily_search_docs:
            tavily_formatted.append(f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>')
        tavily_formatted_search_docs = "\n\n---\n\n".join(tavily_formatted)
        
        # Format Wikipedia results
        wiki_formatted = []
        for doc in wikipedia_search_docs:
            wiki_formatted.append(f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>')
        wikipedia_formatted_search_docs = "\n\n---\n\n".join(wiki_formatted)
        
        # Combine results
        formatted_search_docs = "\n\n---\n\n".join([d for d in [tavily_formatted_search_docs, wikipedia_formatted_search_docs] if d])
        
        # Use TRUSTCALL_INSTRUCTION approach like other memory functions
        TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))
        
        # Initialize spy for Trustcall tool calls
        spy = Spy()
        
        # Create a web search knowledge extractor
        web_search_extractor = create_extractor(
            model,
            tools=[{
                "type": "function",
                "function": {
                    "name": "WebSearchKnowledge",
                    "description": "Tool to process and store web search results while maintaining English tutor persona",
                    "parameters": WebSearchKnowledge.model_json_schema()
                }
            }],
            tool_choice="WebSearchKnowledge",
            enable_inserts=True
        ).with_listeners(on_end=spy)
        
        # Create additional context for the model with search results
        knowledge_msg = HumanMessage(content=f"The user asked: '{question}'. Here is information from web search:\n\n{formatted_search_docs}\n\nProcess this information to answer the question while maintaining your role as an English tutor. Focus on both providing accurate information and using this as a teaching opportunity.")
        
        # Invoke the extractor with combined messages
        result = web_search_extractor.invoke({"messages": updated_messages + [knowledge_msg]})
        
        # Store the synthesized information
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace,
                    rmeta.get("json_doc_id", str(uuid.uuid4())),
                    r.model_dump(mode="json"),
                )
        
        # Get the tool call ID
        tool_calls = state['messages'][-1].tool_calls
        tool_call_id = tool_calls[0]['id'] if tool_calls and len(tool_calls) > 0 else "unknown"
        
        # Extract changes made by Trustcall
        web_search_update_msg = extract_tool_info(spy.called_tools, "WebSearchKnowledge")
        return {"messages": [{"role": "tool", "content": web_search_update_msg or f"Web search completed for: '{question}'", "tool_call_id": tool_call_id}]}
        
    except Exception as e:
        print(f"Error during web search: {str(e)}")
        
        # Get the tool call ID
        tool_calls = state['messages'][-1].tool_calls
        tool_call_id = tool_calls[0]['id'] if tool_calls and len(tool_calls) > 0 else "unknown"
        
        return {"messages": [{"role": "tool", "content": f"Error during web search: {str(e)}", "tool_call_id": tool_call_id}]}
# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_topic", "update_grammar", "update_profile", "web_search_api"]: # type: ignore

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    
    # Verificar se há tool_calls e se não está vazio
    if not hasattr(message, 'tool_calls') or not message.tool_calls:
        print("DEBUG: Nenhuma tool_call encontrada na mensagem, retornando END")
        return END
    
    try:
        tool_call = message.tool_calls[0]
        
        # Verificar se args existe no tool_call
        if 'args' not in tool_call:
            print(f"DEBUG: 'args' não encontrado em tool_call: {tool_call}")
            return END
        
        # Obter o update_type com segurança
        update_type = tool_call['args'].get('update_type')
        print(f"DEBUG: update_type = {update_type}")
        
        if update_type == "profile":
            return "update_profile"
        elif update_type == "topic":
            return "update_topic"
        elif update_type == "grammar":
            return "update_grammar"
        elif update_type == "web_search":
            return "web_search_api"
        else:
            print(f"DEBUG: Tipo de atualização desconhecido: {update_type}, ferramenta completa: {tool_call}")
            return END
    
    except Exception as e:
        print(f"DEBUG: Erro ao processar route_message: {str(e)}")
        return END

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(ai_language_tutor)
builder.add_node(update_topic)
builder.add_node(update_profile)
builder.add_node(update_grammar)
builder.add_node(web_search_api)

# Define the flow 
builder.add_edge(START, "ai_language_tutor")
builder.add_conditional_edges("ai_language_tutor", route_message)
builder.add_edge("update_topic", "ai_language_tutor")
builder.add_edge("update_profile", "ai_language_tutor")
builder.add_edge("update_grammar", "ai_language_tutor")
builder.add_edge("web_search_api", "ai_language_tutor")

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# We compile the graph with the checkpointer and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

file_name = "graph_mermaid.txt"
with open(file_name, "w") as f:
    f.write(graph.get_graph(xray=1).draw_mermaid())

def main():
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory 
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}
    
    input_messages = [HumanMessage(content="Hello, My name is Lance. I live in SF with my wife. I have a 1 year old daughter and I like music and sports.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    ##############################    
    # User input for a topic
    input_messages = [HumanMessage(content="I like jazz music and some pop. I listen often when I working. It help me concentrate.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        
    # Check for updated instructions
    user_id = "Lance"

    # Search 
    for memory in across_thread_memory.search(("grammar", user_id)):
        print(memory.value)
            
    
    #Now we can create a new thread. This creates a new session. Profile, topics, and Instructions saved to long-term memory are accessed. 
    
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory 
    config = {"configurable": {"thread_id": "2", "user_id": "Lance"}}

    # Chat with the chatbot
    input_messages = [HumanMessage(content="Hello, the Lakers win tonight, I exit my job earlier to watch with my wife")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        
    # Test web search function
    input_messages = [HumanMessage(content="Can you tell me about latest advances in quantum computing?")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        
    # Print memory contents
    print("\n=== PROFILE MEMORY ===")
    for memory in across_thread_memory.search(("profile", user_id)):
        print(memory.value)   
    
    print("\n=== WEB SEARCH MEMORY ===")
    for memory in across_thread_memory.search(("web_search", user_id)):
        print(memory.value)         
    
    print("\n=== TOPIC MEMORY ===")
    for memory in across_thread_memory.search(("topic", user_id)):
        print(memory.value)        
        
# if __name__ == "__main__":
#     main()