import uuid
from datetime import datetime
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import MessagesState

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore

from .models import Profile
from .utils import MODEL_SYSTEM_MESSAGE, UpdateMemory
from .service import model, redis_client

def conversational_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Main agent node that handles conversation with the user."""
    
    # Get the user ID from the config
    user_id = config.get("configurable", {}).get("user_id", "default-user")
    
    # Get user profile from Redis
    user_profile = None
    if redis_client:
        try:
            profile_key = f"profile:{user_id}"
            profile_data = redis_client.get(profile_key)
            if profile_data:
                import json
                user_profile = json.loads(profile_data)
        except Exception as e:
            logging.error(f"Error retrieving profile from Redis: {str(e)}")
    
    # Get recent conversations from Redis
    recent_conversations = ""
    if redis_client:
        try:
            pattern = f"conversation:{user_id}:*"
            conversation_keys = redis_client.keys(pattern)
            conversation_keys.sort(reverse=True)
            recent_keys = conversation_keys[:5]  # Get 5 most recent conversations
            
            conversations = []
            for key in recent_keys:
                conversation_data = redis_client.get(key)
                if conversation_data:
                    conversations.append(json.loads(conversation_data))
            
            # Format conversations for the system message
            recent_conversations = "\n".join([
                f"[{conv['timestamp']}]\nUser: {conv['user_message']}\nAssistant: {conv['agent_response']}"
                for conv in conversations
            ])
        except Exception as e:
            logging.error(f"Error retrieving conversations from Redis: {str(e)}")
    
    # Format the system message with user profile and recent conversations
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile if user_profile else "No profile information yet.",
        recent_conversations=recent_conversations if recent_conversations else "No previous conversations."
    )

    # Call the model with the system message and chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Update the user profile based on conversation history."""
    
    # Get the user ID from the config
    user_id = config.get("configurable", {}).get("user_id", "default-user")
    
    try:
        # Extract profile information from conversation
        from .utils import extract_profile_from_conversation
        
        # Use all messages except the latest tool call for extraction
        conversation_history = state["messages"][:-1]
        
        # Extract profile
        profile = extract_profile_from_conversation(model, conversation_history)
        
        # Store in Redis
        if redis_client:
            profile_key = f"profile:{user_id}"
            redis_client.set(profile_key, profile.model_dump_json())
            logging.info(f"Updated profile for user {user_id}")
        
        # Return tool message with verification
        tool_calls = state['messages'][-1].tool_calls
        return {"messages": [{"role": "tool", "content": "Updated user profile", "tool_call_id": tool_calls[0]['id']}]}
    except Exception as e:
        logging.error(f"Error updating profile: {str(e)}")
        tool_calls = state['messages'][-1].tool_calls
        return {"messages": [{"role": "tool", "content": f"Error updating profile: {str(e)}", "tool_call_id": tool_calls[0]['id']}]}

def route_action(state: MessagesState, config: RunnableConfig, store: BaseStore) -> str:
    """Route to appropriate action based on tool call."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    
    tool_call = message.tool_calls[0]
    if tool_call['args']['update_type'] == "profile":
        return "update_profile"
    else:
        logging.error(f"Invalid update type: {tool_call['args']['update_type']}")
        return END

def create_agent_graph():
    """Create and compile the agent graph."""
    try:
        # Create the graph with MessagesState
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("conversational_agent", conversational_agent)
        builder.add_node("update_profile", update_profile)
        
        # Define the flow
        builder.add_edge(START, "conversational_agent")
        builder.add_conditional_edges("conversational_agent", route_action)
        builder.add_edge("update_profile", "conversational_agent")
        
        # Compile the graph
        return builder.compile()
    except Exception as e:
        logging.error(f"Error creating agent graph: {str(e)}")
        raise

# Initialize the agent graph
agent_graph = create_agent_graph()