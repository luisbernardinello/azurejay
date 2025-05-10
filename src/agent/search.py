import logging
from langchain_core.runnables.config import RunnableConfig
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun


from .models import EnhancedState

def search_web(state: EnhancedState, config: RunnableConfig):
    """Retrieve docs from web search"""
    
    # Extract question from the last message
    last_message = state["messages"][-1]
    question = last_message.content
    
    # Initialize search tool here (following DI principle)
    tavily_search = TavilySearchResults(max_results=1)
    
    # Search
    search_docs = tavily_search.invoke(question)
    
    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    logging.info(f"Web search results: {formatted_search_docs}")
    return {"context": [formatted_search_docs]}

def search_wikipedia(state: EnhancedState, config: RunnableConfig):
    """Retrieve docs from wikipedia"""
    
    # Extract question from the last message
    last_message = state["messages"][-1]
    question = last_message.content
    
    try:
        # Search
        search_docs = WikipediaLoader(query=question, load_max_docs=1).load()
        
        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        logging.info(f"Wikipedia search results: {formatted_search_docs}")
        return {"context": [formatted_search_docs]}
    except Exception as e:
        logging.warning(f"Wikipedia search failed: {str(e)}")
        return {"context": ["No relevant Wikipedia results found."]}