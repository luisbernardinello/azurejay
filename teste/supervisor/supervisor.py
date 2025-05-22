from typing import Annotated, Sequence, List, Literal 
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults 
from langgraph.types import Command 
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent 
from IPython.display import Image, display 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
load_dotenv(override=True)

llm = ChatGroq(model="llama-3.3-70b-versatile")

web_search = TavilySearchResults(max_results=2)


research_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    name="research_expert",
    prompt= "You are an Information Specialist with expertise in comprehensive research. Your responsibilities include:\n\n"
            "1. Identifying key information needs based on the query context\n"
            "2. Gathering relevant, accurate, and up-to-date information from reliable sources\n"
            "3. Organizing findings in a structured, easily digestible format\n"
            "4. Citing sources when possible to establish credibility\n"
            "5. Focusing exclusively on information gathering - avoid analysis or implementation\n\n"
            "Provide thorough, factual responses without speculation where information is unavailable."
)

@task
def correction_query(messages):
    """Corrects the grammar in user queries to be more fluent in English."""
    system_message = {
        "role": "system", 
        "content": "You are a Grammar Correction Specialist. Your sole responsibility is to correct grammatical errors in English text.\n\n"
            "Instructions:\n"
            "1. Only correct grammar, spelling, and punctuation errors\n"
            "2. Preserve the original meaning and intent completely\n"
            "3. If the text has NO grammatical errors, respond with exactly: CORRECT\n"
            "4. If the text has errors, respond with ONLY the corrected version\n"
            "5. Do NOT add explanations, greetings, or additional text\n"
            "6. Do NOT engage in conversation - focus solely on correction\n\n"
            "Examples:\n"
            "Input: 'I like jazz music and some pop. I listen often when I working.'\n"
            "Output: 'I like jazz music and some pop. I listen often when I am working.'\n\n"
            "Input: 'Hello, how are you?'\n"
            "Output: CORRECT"
    }
    msg = llm.invoke([system_message] + messages)
    return msg

@entrypoint()
def correction_agent(state):
    correction = correction_query(state['messages']).result()
    messages = add_messages(state["messages"], [correction])
    return {"messages": messages}

correction_agent.name = "correction_agent"

# Enhance Agent
@task
def enhance_query(messages):
    """Enhances user queries to be more specific and actionable."""
    system_message = {
        "role": "system", 
        "content": "You are a Query Refinement Specialist with expertise in transforming vague requests into precise instructions. Your responsibilities include:\n\n"
            "1. Analyzing the original query to identify key intent and requirements\n"
            "2. Resolving any ambiguities without requesting additional user input\n"
            "3. Expanding underdeveloped aspects of the query with reasonable assumptions\n"
            "4. Restructuring the query for clarity and actionability\n"
            "5. Ensuring all technical terminology is properly defined in context\n\n"
            "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible."
    }
    msg = llm.invoke([system_message] + messages)
    return msg

@entrypoint()
def enhancer_agent(state):
    enhanced = enhance_query(state['messages']).result()
    messages = add_messages(state["messages"], [enhanced])
    return {"messages": messages}

enhancer_agent.name = "enhancer_agent"

@task
def validate_response(messages):
    """Validates the quality of responses."""
    system_message = {
        "role": "system", 
        "content": '''You are a Response Validator. Your task is to verify if grammatical corrections have been properly applied.

        Instructions:
        1. Compare the FIRST message (user's original input) with the LAST message (final response)
        2. Check if the original message had grammatical errors
        3. Check if those errors were corrected in the final response
        4. If the original message had errors AND they weren't corrected in the final response, respond with: supervisor
        5. In all other cases, respond with: FINISH

        Validation Logic:
        - Original has errors + Final corrected them = FINISH
        - Original has errors + Final didn't correct them = supervisor  
        - Original has no errors = FINISH
        - If user asked a question and got an answer = FINISH

        Respond with only one word: either "supervisor" or "FINISH"'''
    }
    msg = llm.invoke([system_message] + messages)
    return msg

@entrypoint()
def validator_agent(state):
    validation = validate_response(state['messages']).result()
    messages = add_messages(state["messages"], [validation])
    return {"messages": messages}

validator_agent.name = "validator_agent"


supervisor_node = create_supervisor(
    model = llm,
    agents=[research_agent, enhancer_agent, validator_agent, correction_agent],
    prompt=(
        "You are a friendly supervisor who manages a team of specialized agents and converses with users.\n\n"
        "Your workflow:\n"
        "1. ALWAYS start by calling the correction_agent to check for grammar errors\n"
        "2. ONLY if the user's message contains a direct question (indicated by question marks, interrogative words like what, why, how, etc.) or explicitly requests information, call the enhancer_agent to improve query clarity\n"
        "3. After the enhancer_agent (if called), for research or information needs, call the research_agent\n"
        "4. ALWAYS finish by calling the validator_agent to verify corrections were applied\n\n"
        "Conversation Guidelines:\n"
        "- Respond to users in a warm, friendly manner as a long-term friend\n"
        "- Use the last message of validator_agent to respond to the user\n"
        "- Provide helpful, conversational responses\n"
        "- Acknowledge the user's input naturally\n"
        "- If corrections were made, you must mention them casually\n\n"
        "Agent Assignments:\n"
        "- correction_agent: For grammar checking (always call first)\n"
        "- enhancer_agent: ONLY when user asks questions or requests information\n"
        "- research_agent: For information gathering needs (ONLY after enhancer_agent if a question was asked)\n"
        "- validator_agent: Always call last to verify workflow\n\n"
        "**Important limitations**:\n\n"
        "1. You do NOT evaluate the quality of any agent's work\n"
        "2. You do NOT provide feedback on any agent's output\n"
        "3. You do NOT modify any agent's content\n"
        "Work with one agent at a time, do not call agents in parallel.\n\n"
        "CRITICAL: For messages that are statements (not questions), follow this workflow only:\n"
        "1. Call correction_agent\n"
        "2. Call validator_agent\n"
        "3. Respond to the user\n"
        "DO NOT call enhancer_agent or research_agent for statements."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
    # output_mode="last_message",
    parallel_tool_calls=False,
)

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

checkpointer = InMemorySaver()
store = InMemoryStore()
graph = supervisor_node.compile(checkpointer=within_thread_memory, store=across_thread_memory)



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
        if "messages" in chunk and chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    # User mentions music preferences with grammar errors
    input_messages = [HumanMessage(content="I like jazz music and some pop. I listen often when I working. It help me concentrate.")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages, "step_count": 0, "memory_updates": []}, config, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            chunk["messages"][-1].pretty_print()
        
    # Check for updated grammar corrections
    user_id = "Lance"
    for memory in across_thread_memory.search(("correction", user_id)):
        print(memory.value)
    
    # Create a new thread with access to long-term memory
    config = {"configurable": {"thread_id": "2", "user_id": "Lance"}}

    # User mentions sports with grammar errors
    input_messages = [HumanMessage(content="Hello, the Lakers win tonight, I exit my job earlier to watch with my wife")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages, "step_count": 0, "memory_updates": []}, config, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            chunk["messages"][-1].pretty_print()
        
if __name__ == "__main__":
    main()