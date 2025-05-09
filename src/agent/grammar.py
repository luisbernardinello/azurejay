import logging
import spacy
from textblob import TextBlob
import re
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, List, Tuple, Optional
from langdetect import detect, LangDetectException

from . import models
from . import utils

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logging.warning("English spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def detect_language(text: str) -> str:
    """
    Detect the language of the input text using langdetect
    """
    try:
        # Use a more substantial sample of text
        sample = text[:100] if len(text) > 100 else text
        return detect(sample)
    except LangDetectException as e:
        logging.warning(f"Language detection failed: {e}")
        return "unknown"  # Default to unknown rather than assuming English

def check_grammar(text: str) -> Tuple[List[str], Optional[str]]:
    """
    Check the grammar of the text using spaCy and TextBlob
    Returns a list of issues and optionally a suggestion for correction
    """
    issues = []
    correction_suggestion = None

    if not nlp:
        return issues, correction_suggestion

    # Use TextBlob for spell checking and basic corrections
    blob = TextBlob(text)
    
    # Check for spelling errors (only in English)
    lang = detect_language(blob.raw)
    if lang == 'en':
        # Find words that might be misspelled
        misspelled_words = []
        for word in blob.words:
            if word.isalpha() and len(word) > 2:  # Only check actual words (not numbers, symbols)
                if word.lower() not in nlp.vocab:
                    corrections = word.spellcheck()
                    if corrections and corrections[0][1] > 0.8:  # Only suggest if confidence is high
                        misspelled_words.append((word, corrections[0][0]))
        
        if misspelled_words:
            for original, suggestion in misspelled_words:
                issues.append(f"Possible spelling error: '{original}' -> '{suggestion}'")
    
    # Use spaCy for more complex grammar checks (only for English)
    if lang == 'en':
        doc = nlp(text)
        
        # Check for missing punctuation at the end of sentences
        sentences = list(doc.sents)
        for sent in sentences:
            if sent.text.strip() and not re.search(r'[.!?]$', sent.text.strip()):
                issues.append("Missing end punctuation in sentence: " + sent.text.strip())

        # Generate correction suggestion if there are issues
        if issues:
            # Apply simple corrections
            corrected_text = text
            for original, suggestion in misspelled_words:
                # Simple word replacement - this is basic and could be improved
                corrected_text = re.sub(r'\b' + re.escape(original) + r'\b', suggestion, corrected_text)
            
            # Add period if missing at the end
            if corrected_text and not re.search(r'[.!?]$', corrected_text.strip()):
                corrected_text = corrected_text.strip() + "."
                
            correction_suggestion = corrected_text
    
    return issues, correction_suggestion

def needs_grammar_correction(text: str) -> bool:
    """
    Determine if a text needs grammar correction or language detection
    Returns True if the text is not in English or has significant issues
    """
    if not text:
        return False
    
    # Detect language first - this is the key check
    language = detect_language(text)
    
    # If not English, we should respond with a language notice
    if language != 'en' and language != 'unknown':
        return True
    
    # Only process English text for grammar issues
    if language == 'en':
        issues, _ = check_grammar(text)
        # Only suggest corrections if there are meaningful issues
        return len(issues) > 0
    
    return False

def process_grammar(state: models.EnhancedState) -> Dict[str, Any]:
    """
    Process the grammar of the last user message
    Returns the grammar issues and corrected text if applicable
    """
    # Extract the last user message
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return {"grammar_issues": None, "corrected_text": None}
    
    text = last_message.content
    
    # Detect language - improved with better error handling
    language = detect_language(text)
    
    # If not English, return with language indication
    if language != 'en' and language != 'unknown':
        return {
            "grammar_issues": {
                "language_detected": language,
                "issues": ["Non-English text detected"],
                "needs_response": True
            },
            "corrected_text": None
        }
    
    # If language is unknown but text exists, we'll try to process as English
    if language == 'unknown' and text:
        logging.warning(f"Language detection uncertain for text: '{text[:50]}...'")
        # Default to English for unknown but attempt grammar check
        language = 'en'
    
    # Check grammar for English text
    issues, corrected_text = check_grammar(text)
    
    # Only return grammar issues if significant ones are found
    if issues:
        return {
            "grammar_issues": {
                "language_detected": language,
                "issues": issues,
                "needs_response": True
            },
            "corrected_text": corrected_text
        }
    
    return {"grammar_issues": None, "corrected_text": None}

def generate_grammar_response(state: models.EnhancedState, config: RunnableConfig, components: dict) -> Dict[str, Any]:
    """
    Generate a response about grammar issues for the user
    """
    # Get grammar issues from state
    grammar_issues = state.get("grammar_issues")
    corrected_text = state.get("corrected_text")
    
    if not grammar_issues or not grammar_issues.get("needs_response"):
        # No grammar issues to handle, this should not happen
        return state
    
    # Create a prompt for the LLM to generate a helpful grammar correction
    language_detected = grammar_issues.get("language_detected", "unknown")
    issues = grammar_issues.get("issues", [])
    
    if language_detected != "en" and language_detected != "unknown":
        # Create a system message for non-English detection
        system_msg = """You are a helpful assistant. The user has sent a message in a language that's not English. 
        Respond ONLY in English that you currently only understand English. 
        If you can recognize the language, mention it in your response. 
        Be friendly and apologetic."""
        
        prompt = f"The user sent a message in '{language_detected}'. Respond politely IN ENGLISH ONLY that you only understand English."
    else:
        # Create a system message for grammar correction
        system_msg = """You are a helpful assistant that can provide gentle grammar corrections.
        When users make grammatical errors, you should politely point them out and suggest corrections.
        Be kind and educational in your approach, explaining briefly why the correction is needed.
        Only focus on errors that would hinder communication or understanding."""
        
        # Format the grammar issues into a prompt
        issues_text = "\n".join([f"- {issue}" for issue in issues])
        prompt = f"""The user's message had the following potential grammar issues:

{issues_text}

Original text: "{state['question']}"
Suggested correction: "{corrected_text if corrected_text else state['question']}"

Provide a polite and helpful response about these grammar issues. 
Be gentle and educational. Don't just correct; explain briefly why.
Then ask if they want to continue with the conversation using the corrected text.
Keep your response brief and friendly."""

    # Generate a response about the grammar issues
    response = components["model"].invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=prompt)
    ])
    
    # Return the state with the grammar response
    return {
        "messages": state["messages"] + [response],
        "answer": response.content,
        "grammar_correction": models.GrammarCorrection(
            original_text=state["question"],
            corrected_text=corrected_text if corrected_text else state["question"],
            issues_found=issues,
            language_detected=language_detected
        )
    }