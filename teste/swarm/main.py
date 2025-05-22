from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from langgraph_swarm import SwarmState, add_active_agent_router
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Import the updated agents
from correction import correction_graph
from profile import profile_graph

load_dotenv(override=True)

# Initialize memory stores
across_thread_memory = InMemoryStore()
within_thread_memory = MemorySaver()
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Create the swarm workflow following the documentation exactly
workflow = (
    StateGraph(SwarmState)
    .add_node("Correction", correction_graph, destinations=("Profile",))
    .add_node("Profile", profile_graph, destinations=("Correction",))
)

# Add the active agent router
workflow = add_active_agent_router(
    builder=workflow,
    route_to=["Correction", "Profile"],
    default_active_agent="Correction",  # Start with Correction agent as default
)

def main():
    # Compile the workflow
    graph = workflow.compile(checkpointer=within_thread_memory, store=across_thread_memory)
    
    # Save the graph visualization
    file_name = "graph_mermaid.txt"
    with open(file_name, "w") as f:
        f.write(graph.get_graph(xray=1).draw_mermaid())
        
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}

    print("=== Test 1: User Introduction (Should check grammar first, then extract profile) ===")
    # User introduction
    input_messages = [HumanMessage(content="Hello, My name is Lance. I live in SF with my wife. I have a 1 year old daughter and I like music and sports.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        if chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    print("\n=== Test 2: User with Grammar Errors (Should correct first) ===")
    # User mentions music preferences with grammar errors
    input_messages = [HumanMessage(content="I like jazz music and some pop. I listen often when I working. It help me concentrate.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        if chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    print("\n=== Test 3: New Thread with Grammar Errors ===")
    # Create a new thread with access to long-term memory
    config = {"configurable": {"thread_id": "2", "user_id": "Lance"}}

    # User mentions sports with grammar errors
    input_messages = [HumanMessage(content="Hello, the Lakers win tonight, I exit my job earlier to watch with my wife")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        if chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    print("\n=== Test 4: Correct Grammar (Should just continue conversation) ===")
    # User with correct grammar
    input_messages = [HumanMessage(content="How are you today? I hope you're doing well.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        if chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    print("\n=== FINAL MEMORY CHECK ===")
    user_id = "Lance"
    
    print("\n=== PROFILE MEMORY ===")
    for memory in across_thread_memory.search(("memory", user_id)):
        print(memory.value)   
        
    print("\n=== GRAMMAR CORRECTION MEMORY ===")
    for memory in across_thread_memory.search(("corrections", user_id)):
        print(memory.value)   
        
if __name__ == "__main__":
    main()