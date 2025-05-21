from typing import Annotated, Sequence, List, Literal 
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults 
from langgraph.types import Command 
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent 
from IPython.display import Image, display 
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv(override=True)

# Use Groq LLM as in the second example
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Initialize search tool
web_search = TavilySearchResults(max_results=2)

# Python REPL tool
python_repl_tool = PythonREPLTool()

# Initialize memory storage for user interaction history
checkpointer = InMemorySaver()
store = InMemoryStore()

class Supervisor(BaseModel):
    next: Literal["correction", "enhancer", "researcher"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
                    "'correction' for grammar checking (should always be called first), "
                    "'enhancer' when user queries need improvement for clarity, "
                    "'researcher' when information gathering is needed."
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )

def supervisor_node(state: MessagesState) -> Command[Literal["correction", "enhancer", "researcher"]]:
    """
    Supervisor node that manages workflow between specialized agents.
    Routes tasks to appropriate specialists based on the current state.
    """
    system_prompt = '''
    You are a workflow supervisor managing a team of specialized agents. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task. Provide a clear, concise rationale for each decision to ensure transparency in your decision-making process.

    **Team Members**:
    1. **Correction Agent**: Always consider this agent first. They check and correct grammar errors in English text.
    2. **Enhancer Agent**: They clarify ambiguous requests, improve poorly defined queries, and ensure the task is well-structured before deeper processing begins.
    3. **Researcher Agent**: Specializes in information gathering, fact-finding, and collecting relevant data needed to address the user's request.

    **Your Responsibilities**:
    1. Analyze each user request and agent response for completeness, accuracy, and relevance.
    2. Route the task to the most appropriate agent at each decision point.
    3. Maintain workflow momentum by avoiding redundant agent assignments.
    4. Continue the process until the user's request is fully and satisfactorily resolved.

    Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps.
    '''
    
    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"] 

    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")
    
    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,  
    )

def correction_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Grammar correction node that checks and corrects English grammar errors.
    Takes the current message and corrects grammatical mistakes.
    """
    system_prompt = '''
    You are a Grammar Correction Specialist. Your sole responsibility is to correct grammatical errors in English text.

    Instructions:
    1. Only correct grammar, spelling, and punctuation errors
    2. Preserve the original meaning and intent completely
    3. If the text has NO grammatical errors, respond with exactly: CORRECT
    4. If the text has errors, respond with ONLY the corrected version
    5. Do NOT add explanations, greetings, or additional text
    6. Do NOT engage in conversation - focus solely on correction

    Examples:
    Input: 'I like jazz music and some pop. I listen often when I working.'
    Output: 'I like jazz music and some pop. I listen often when I am working.'

    Input: 'Hello, how are you?'
    Output: CORRECT
    '''

    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"]

    correction_response = llm.invoke(messages)

    print(f"--- Workflow Transition: Correction → Validator ---")

    return Command(
        update={
            "messages": [
                HumanMessage(content=correction_response.content, name="correction")
            ]
        },
        goto="validator", 
    )
    
def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer agent node that improves and clarifies user queries.
    Takes the original user input and transforms it into a more precise,
    actionable request before passing it to the supervisor.
    """

    system_prompt = (
        "You are a Query Refinement Specialist with expertise in transforming vague requests into precise instructions. Your responsibilities include:\n\n"
        "1. Analyzing the original query to identify key intent and requirements\n"
        "2. Resolving any ambiguities without requesting additional user input\n"
        "3. Expanding underdeveloped aspects of the query with reasonable assumptions\n"
        "4. Restructuring the query for clarity and actionability\n"
        "5. Ensuring all technical terminology is properly defined in context\n\n"
        "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible."
    )

    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"]  

    enhanced_query = llm.invoke(messages)

    print(f"--- Workflow Transition: Enhancer → Supervisor ---")

    return Command(
        update={
            "messages": [  
                HumanMessage(
                    content=enhanced_query.content, 
                    name="enhancer"  
                )
            ]
        },
        goto="supervisor", 
    )
    
def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research agent node that gathers information using web search.
    Takes the current task state, performs relevant research,
    and returns findings for validation.
    """
    
    research_agent = create_react_agent(
        llm,  
        tools=[web_search],  
        state_modifier="You are an Information Specialist with expertise in comprehensive research. Your responsibilities include:\n\n"
            "1. Identifying key information needs based on the query context\n"
            "2. Gathering relevant, accurate, and up-to-date information from reliable sources\n"
            "3. Organizing findings in a structured, easily digestible format\n"
            "4. Citing sources when possible to establish credibility\n"
            "5. Focusing exclusively on information gathering - avoid analysis or implementation\n\n"
            "Provide thorough, factual responses without speculation where information is unavailable."
    )

    result = research_agent.invoke(state)

    print(f"--- Workflow Transition: Researcher → Validator ---")

    return Command(
        update={
            "messages": [ 
                HumanMessage(
                    content=result["messages"][-1].content,  
                    name="researcher"  
                )
            ]
        },
        goto="validator", 
    )

class Validator(BaseModel):
    next: Literal["supervisor", "responder"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue processing or 'responder' when ready for final response."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

def validator_node(state: MessagesState) -> Command[Literal["supervisor", "responder"]]:
    """
    Validates the quality of responses and determines if the workflow should continue or move to final response.
    """
    system_prompt = '''
    You are a Response Validator. Your task is to verify if grammatical corrections have been properly applied.

    Instructions:
    1. Compare the FIRST message (user's original input) with the LAST message (final response)
    2. Check if the original message had grammatical errors
    3. Check if those errors were corrected in the final response
    4. If the original message had errors AND they weren't corrected in the final response, respond with: supervisor
    5. In all other cases, respond with: responder

    Validation Logic:
    - Original has errors + Final corrected them = responder
    - Original has errors + Final didn't correct them = supervisor  
    - Original has no errors = responder
    - If user asked a question and got an answer = responder

    Respond with only one word: either "supervisor" or "responder"
    '''

    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    response = llm.with_structured_output(Validator).invoke(messages)

    goto = response.next
    reason = response.reason

    if goto == "responder":
        print(f"--- Workflow Transition: Validator → Responder ---")
    else:
        print(f"--- Workflow Transition: Validator → Supervisor ---")

    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="validator")
            ]
        },
        goto=goto, 
    )


def responder_node(state: MessagesState) -> Command[Literal["__end__"]]:
    """
    Responder node that provides the final friendly response to the user.
    Takes all the processed information and constructs a warm, conversational response.
    """
    system_prompt = '''
    You are a friendly English language tutor who converses with users.

    Conversation Guidelines:
    - Respond to users in a warm, friendly manner as if you're a long-term friend
    - Provide helpful, conversational responses
    - Acknowledge the user's input naturally
    - If you notice corrections were made to their English, mention them casually and constructively
    - Be encouraging about their language learning journey
    - Keep your tone positive and supportive

    Important: You are the final touchpoint with the user, so make sure your response is complete and helpful.
    '''

    # Get the original user message and the most recent correction/enhancement
    user_message = state["messages"][0].content
    
    # Find the most recent correction or research response
    correction_message = None
    research_message = None
    
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'name') and msg.name == "correction" and not correction_message:
            correction_message = msg.content
        if hasattr(msg, 'name') and msg.name == "researcher" and not research_message:
            research_message = msg.content
            
    conversation_context = f"User said: {user_message}\n\n"
    
    if correction_message and correction_message != "CORRECT":
        conversation_context += f"Grammar correction: {correction_message}\n\n"
    
    if research_message:
        conversation_context += f"Research findings: {research_message}\n\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation_context}
    ]

    final_response = llm.invoke(messages)

    print(f"--- Workflow Transition: Responder → END ---")

    return Command(
        update={
            "messages": [
                HumanMessage(content=final_response.content, name="responder")
            ]
        },
        goto=END,
    )
    
    
# Initialize the graph with message state
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("supervisor", supervisor_node)
graph.add_node("correction", correction_node)
graph.add_node("enhancer", enhancer_node)
graph.add_node("researcher", research_node)
graph.add_node("validator", validator_node)
graph.add_node("responder", responder_node)

# Add edges
graph.add_edge(START, "supervisor")
app = graph.compile(checkpointer=checkpointer, store=store)

# Display the graph
file_name = "graph_mermaid.txt"
with open(file_name, "w") as f:
    f.write(app.get_graph(xray=1).draw_mermaid())
    
def main():
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory 
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}
    
    # User introduction
    # input_messages = {
    #     "messages": [
    #         HumanMessage(content="Hello, My name is Lance. I live in SF with my wife. I have a 1 year old daughter and I like music and sports.")
    #     ]
    # }
    input_messages = {
        "messages": [
            HumanMessage(content="Hello, how are you?")
        ]
    }

    # Run the graph
    print("\n=== Testing User Introduction ===\n")
    for event in app.stream(input_messages, config):
        for key, value in event.items():
            if value is None:
                continue
            last_message = value.get("messages", [])[-1] if "messages" in value else None
            if last_message:
                print(f"Output from node '{key}':")
                print(f"{last_message.name}: {last_message.content}")
                print()
    
    # # User mentions music preferences with grammar errors
    # input_messages = {
    #     "messages": [
    #         HumanMessage(content="I like jazz music and some pop. I listen often when I working. It help me concentrate.")
    #     ]
    # }

    # # Run the graph
    # print("\n=== Testing Grammar Correction ===\n")
    # for event in app.stream(input_messages, config):
    #     for key, value in event.items():
    #         if value is None:
    #             continue
    #         last_message = value.get("messages", [])[-1] if "messages" in value else None
    #         if last_message:
    #             print(f"Output from node '{key}':")
    #             print(f"{last_message.name}: {last_message.content}")
    #             print()
        
    # # Create a new thread with access to long-term memory
    # config = {"configurable": {"thread_id": "2", "user_id": "Lance"}}

    # # User mentions sports with grammar errors
    # input_messages = {
    #     "messages": [
    #         HumanMessage(content="Hello, the Lakers win tonight, I exit my job earlier to watch with my wife")
    #     ]
    # }

    # # Run the graph
    # print("\n=== Testing Memory Access Across Threads ===\n")
    # for event in app.stream(input_messages, config):
    #     for key, value in event.items():
    #         if value is None:
    #             continue
    #         last_message = value.get("messages", [])[-1] if "messages" in value else None
    #         if last_message:
    #             print(f"Output from node '{key}':")
    #             print(f"{last_message.name}: {last_message.content}")
    #             print()

if __name__ == "__main__":
    main()