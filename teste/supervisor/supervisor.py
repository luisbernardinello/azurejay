from typing import Annotated, Sequence, List, Literal 
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults 
from langgraph.types import Command 
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent 
from IPython.display import Image, display 
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

# Import the responder subgraph
from responder import create_responder_subgraph
from language_tool import LanguageToolAPI
import configuration

load_dotenv(override=True)

# Use Groq LLM as in the second example
llm = ChatGroq(model="llama-3.3-70b-versatile")
language_tool = LanguageToolAPI(base_url="https://api.languagetool.org/v2")
llm_verifier = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# Initialize search tool
web_search = TavilySearchResults(max_results=2)

# Python REPL tool
python_repl_tool = PythonREPLTool()

# Initialize memory storage for user interaction history
checkpointer = InMemorySaver()
store = InMemoryStore()

class Supervisor(BaseModel):
    next: Literal["correction", "researcher"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
                    "'correction' for grammar checking (should always be called first), "
                    "'researcher' when information gathering is needed."
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )

def get_last_user_message(state: MessagesState) -> str:
    """Extract the last user message (not from agents)"""
    last_user_message = next(
        (msg.content for msg in reversed(state["messages"]) 
         if isinstance(msg, HumanMessage) and not hasattr(msg, 'name') or msg.name is None), 
        ""
    )
    return last_user_message

def supervisor_node(state: MessagesState) -> Command[Literal["correction", "researcher"]]:
    """
    Supervisor node that manages workflow between specialized agents.
    Routes tasks to appropriate specialists based on the current state.
    """
    system_prompt = '''
    You are a workflow supervisor managing a team of specialized agents. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task. Provide a clear, concise rationale for each decision to ensure transparency in your decision-making process.
    Work with one agent at a time, do not call agents in parallel.

    #Team Members:
    1. **Correction Agent**: Always consider this agent first. They check and correct grammar errors in English text.
    2. **Researcher Agent**: Specializes in information gathering, fact-finding, and collecting relevant data needed to address the user's request.
    
    #Your workflow:
    1. ALWAYS start by calling the Correction Agent to check for grammar errors
    2. ONLY if the user asks a clear, direct question that requires external information (e.g., 'What is the capital of France?', 'Who won the game last night?'), call the Researcher Agent
        
    #Important limitations
    1. You do NOT evaluate the quality of any agent's work
    2. You do NOT provide feedback on any agent's output
    3. You do NOT modify any agent's content

    CRITICAL: For messages that are statements (not questions), follow this workflow only:
    1. Call Correction Agent
    DO NOT call Researcher Agent for statements.
    
    Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps.
    '''
    
    # Only get the last user message for the supervisor decision
    last_user_msg = get_last_user_message(state)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_user_msg}
    ] 

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

def correction_node(state: MessagesState) -> Command[Literal["responder"]]:
    print("--- Correction Node: Starting ADVANCED check... ---")

    # 1. Get the last user message (not from agents)
    last_user_message_obj = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage) and (not hasattr(msg, 'name') or msg.name is None)), None
    )
    if not last_user_message_obj:
        return Command(update={"messages": [HumanMessage(content="CORRECT", name="correction")]}, goto="responder")
    
    user_message_content = last_user_message_obj.content
    print(f"--- Correction Node: Analyzing user message: '{user_message_content}' ---")

    # --- LAYER 1: LANGUAGETOOL (SYNTAX) ---
    lt_found_errors = False
    lt_corrected_text = user_message_content
    try:
        lt_result = language_tool.check_text(user_message_content)
        if lt_result.errors:
            lt_found_errors = True
            lt_corrected_text = lt_result.corrected_text
            print(f"--- Correction Node: LanguageTool found errors. LT Corrected Text: '{lt_corrected_text}' ---")
        else:
            print("--- Correction Node: LanguageTool found NO syntax errors. ---")
    except Exception as e:
        print(f"--- Correction Node: LanguageTool API failed: {e}. Using original text for next steps. ---")

    # --- LAYER 2: LLM VERIFIER (SEMANTICS) ---
    verifier_made_semantic_correction = False
    text_after_semantic_check = user_message_content

    if user_message_content.strip() and len(user_message_content.split()) > 1:
        print("--- Correction Node: Proceeding to Semantic Correction layer (Verifier LLM). ---")
        try:
            semantic_correction_prompt = f'''
            You are a semantic correction specialist for English language learners.
            Analyze the following sentence: "{user_message_content}"
            1. Identify ONLY subtle semantic or contextual errors (like incorrect verb tense for a past event, wrong word choice for the context, etc.). Do NOT focus on minor punctuation or pure grammatical errors unless they create semantic ambiguity.
            2. If you find semantic errors, provide the sentence corrected ONLY for those semantic errors.
            3. If you find NO semantic errors, respond with the exact string "NO_SEMANTIC_ERRORS_FOUND".
            Respond with ONLY the semantically corrected sentence or "NO_SEMANTIC_ERRORS_FOUND".
            '''
            messages_for_verifier = [
                SystemMessage(content="You are a precise linguistic analyst focusing on semantic corrections."),
                HumanMessage(content=semantic_correction_prompt)
            ]
            verifier_response = llm_verifier.invoke(messages_for_verifier)
            print(f"--- Correction Node: Verifier LLM raw response: '{verifier_response.content}' ---")

            if verifier_response.content.strip().upper() != "NO_SEMANTIC_ERRORS_FOUND" and verifier_response.content.strip() != user_message_content.strip():
                verifier_made_semantic_correction = True
                text_after_semantic_check = verifier_response.content.strip()
                print(f"--- Correction Node: Verifier LLM made a semantic correction: '{text_after_semantic_check}' ---")
            else:
                print("--- Correction Node: Verifier LLM found NO semantic errors or made no changes. ---")
        except Exception as e:
            print(f"--- Correction Node: Verifier LLM failed: {e}. Assuming no semantic correction made by verifier. ---")
    else:
        print("--- Correction Node: Skipping Semantic Correction due to empty or trivial user message. ---")
        
    # --- DECISION LOGIC AND SYNTHESIS ---
    final_correction_content = ""

    if not lt_found_errors and not verifier_made_semantic_correction:
        print("--- Correction Node: No errors/corrections by any tool. Outputting CORRECT. ---")
        final_correction_content = "CORRECT"
    else:
        print("--- Correction Node: Corrections suggested. Performing synthesis with Full LLM. ---")
        
        synthesizer_prompt_system = f'''
        You are an expert English language editor. Your task is to produce a final, perfectly corrected sentence for an English learner, based on the original text and suggestions from other tools.

        Original user sentence: "{user_message_content}"

        Tool 1 (LanguageTool - Syntax) analysis:
        - Detected syntax errors: {"YES" if lt_found_errors else "NO"}
        - LanguageTool's suggested correction (if any): "{lt_corrected_text if lt_found_errors else 'N/A'}"

        Tool 2 (Verifier LLM - Semantics) analysis:
        - Detected/Corrected semantic issues: {"YES" if verifier_made_semantic_correction else "NO"}
        - Verifier LLM's suggested semantic correction (if any): "{text_after_semantic_check if verifier_made_semantic_correction else 'N/A'}"

        Your Task:
        1. Review the Original user sentence.
        2. Consider the corrections suggested by LanguageTool (focused on syntax) and the Verifier LLM (focused on semantics).
        3. Produce ONE single, final, perfectly corrected version of the original sentence that integrates the best of these suggestions and your own expertise.
        4. If, after reviewing all inputs, you believe the Original user sentence was already perfect AND no tools made valid corrections, respond with the exact string "CORRECT".
        5. Otherwise, respond ONLY with the fully corrected sentence.
        '''
        
        # Only use the current message for synthesis, not full history
        messages_for_synthesizer = [
            SystemMessage(content=synthesizer_prompt_system),
            HumanMessage(content=user_message_content)
        ]

        synthesis_response = llm.invoke(messages_for_synthesizer)
        final_correction_content = synthesis_response.content.strip()
        print(f"--- Correction Node: Synthesizer LLM response: '{final_correction_content}' ---")

    print(f"--- Correction Node: Outputting '{final_correction_content}' with name 'correction'. ---")
    return Command(
        update={"messages": [HumanMessage(content=final_correction_content, name="correction")]},
        goto="responder",
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

    # Only get the last user message for enhancement
    last_user_msg = get_last_user_message(state)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_user_msg}
    ]

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
    
def research_node(state: MessagesState) -> Command[Literal["responder"]]:
    """
    Research agent node that gathers information using web search.
    Takes the current task state, performs relevant research,
    and returns findings to responder.
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

    # Get the last relevant message (either from supervisor or user)
    last_relevant_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            if hasattr(msg, 'name') and msg.name == "supervisor":
                last_relevant_message = msg
                break
            elif not hasattr(msg, 'name') or msg.name is None:
                last_relevant_message = msg
                break
    
    # Create a minimal state for the research agent
    research_state = {"messages": [last_relevant_message] if last_relevant_message else state["messages"][-1:]}
    
    result = research_agent.invoke(research_state)

    print(f"--- Workflow Transition: Researcher → Responder ---")

    return Command(
        update={
            "messages": [ 
                HumanMessage(
                    content=result["messages"][-1].content,  
                    name="researcher"  
                )
            ]
        },
        goto="responder", 
    )

def call_responder_subgraph(state: MessagesState, config: RunnableConfig):
    """
    Node function that calls the responder subgraph.
    This transforms the supervisor state to the subgraph state and invokes it.
    """
    # Create the responder subgraph
    responder_subgraph = create_responder_subgraph()
    
    print(f"--- Workflow Transition: Responder Subgraph Started ---")
    
    # Invoke the subgraph with the current state and config
    result = responder_subgraph.invoke(state, config)
    
    print(f"--- Workflow Transition: Responder Subgraph Completed → END ---")
    
    # Return the result as a Command to END
    return Command(
        update={
            "messages": result["messages"]
        },
        goto=END,
    )
    
# Initialize the graph with message state
graph = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Add nodes
graph.add_node("supervisor", supervisor_node)
graph.add_node("correction", correction_node)
# graph.add_node("enhancer", enhancer_node)
graph.add_node("researcher", research_node)
graph.add_node("responder", call_responder_subgraph)

# Add edges
graph.add_edge(START, "supervisor")
app = graph.compile(checkpointer=checkpointer, store=store)


def main():
    # Compile the workflow    
    # Save the graph visualization
    file_name = "graph_mermaid.txt"
    with open(file_name, "w") as f:
        f.write(app.get_graph(xray=1).draw_mermaid())
        
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}

    print("=== Test 1: User Introduction (Should check grammar first, then extract profile) ===")
    # User introduction
    input_messages = [HumanMessage(content="Hello, My name is Lance. I live in SF with my wife. I have a 1 year old daughter and I like music and sports.")]

    # Run the graph
    for chunk in app.stream({"messages": input_messages}, config, stream_mode="values"):
        if chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    print("\n=== Test 2: User with Grammar Errors (Should correct first) ===")
    # User mentions music preferences with grammar errors
    input_messages = [HumanMessage(content="I like jazz music and some pop. I listen often when I working. It help me concentrate.")]

    # Run the graph
    for chunk in app.stream({"messages": input_messages}, config, stream_mode="values"):
        if chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    print("\n=== Test 3: New Thread with Grammar Errors ===")
    # Create a new thread with access to long-term memory
    config = {"configurable": {"thread_id": "2", "user_id": "Lance"}}

    # User mentions sports with grammar errors
    input_messages = [HumanMessage(content="Hello, the Lakers win tonight, I exit my job earlier to watch with my wife")]

    # Run the graph
    for chunk in app.stream({"messages": input_messages}, config, stream_mode="values"):
        if chunk["messages"]:
            chunk["messages"][-1].pretty_print()
    
    print("\n=== FINAL MEMORY CHECK ===")
    user_id = "Lance"
    
    print("\n=== PROFILE MEMORY ===")
    for memory in store.search(("memory", user_id)):
        print(memory.value)   
        
    print("\n=== GRAMMAR CORRECTION MEMORY ===")
    for memory in store.search(("corrections", user_id)):
        print(memory.value)   
        
if __name__ == "__main__":
    main()