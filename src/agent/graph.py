
from dataclasses import dataclass
import uuid
import os
from datetime import datetime
from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Annotated, Literal, Optional, TypedDict, Dict, Any, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.checkpoint.redis import RedisSaver, AsyncRedisSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore, AsyncRedisStore

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

from src.agent import configuration
from src.agent.models import *
from src.agent.utils import *
from src.database.core import get_redis
load_dotenv()

## Node definitions

def ai_language_tutor(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response."""
    model = initialize_model()
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
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=True).invoke([SystemMessage(content=system_msg)]+state["messages"])

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
    profile_extractor = get_profile_extractor()
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
    model = initialize_model()
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
    model = initialize_model()
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
    model = initialize_model()
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
    
    if not hasattr(message, 'tool_calls') or not message.tool_calls:
        print("DEBUG: Nenhuma tool_call encontrada na mensagem, retornando END")
        return END
    
    try:
        tool_call = message.tool_calls[0]
        
        if 'args' not in tool_call:
            print(f"DEBUG: 'args' não encontrado em tool_call: {tool_call}")
            return END
        
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
    
def setup_and_run_graph():
    
    # redis_uri = get_redis()
    REDIS_URI = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    """Setup the graph and run a sample conversation"""
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

    # Initialize Redis connections
    with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
        # Setup indices
        checkpointer.setup()
        
        with RedisStore.from_conn_string(REDIS_URI) as store:
            # Setup store indices
            store.setup()
            
            # Compile the graph with Redis persistence
            graph = builder.compile(checkpointer=checkpointer, store=store)
            
            
            return graph