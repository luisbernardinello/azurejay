import logging
import spacy
from textblob import TextBlob
import re
from spellchecker import SpellChecker
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, List, Tuple, Optional, Set
from langdetect import detect, LangDetectException
import requests

from . import models

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logging.warning("English spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize SpellChecker
spell = SpellChecker()

# LanguageTool API endpoint
LANGUAGE_TOOL_API = "https://api.languagetool.org/v2/check"

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

def check_spelling_with_pyspellchecker(text: str, lang: str = 'en') -> List[Tuple[str, str]]:
    """
    Check spelling using pyspellchecker library
    Returns a list of (misspelled_word, suggested_correction) tuples
    """
    if lang != 'en':
        return []  # PySpellChecker works best with English
        
    # Split the text into words, removing punctuation
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    # Find misspelled words
    misspelled = spell.unknown(words)
    
    # Generate corrections
    corrections = []
    for word in misspelled:
        # Get the most likely correction
        correction = spell.correction(word)
        if correction and correction != word:
            corrections.append((word, correction))
            
    return corrections

def check_with_languagetool(text: str, lang: str = 'en-US') -> List[Dict[str, Any]]:
    """
    Use the LanguageTool API to check for grammar issues
    Returns a list of detailed grammar issues found with full metadata
    """
    if not text:
        return []
        
    try:
        params = {
            'text': text,
            'language': lang,
            'enabledOnly': 'false',
        }
        
        response = requests.post(LANGUAGE_TOOL_API, data=params)
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"LanguageTool found {len(result.get('matches', []))} issues")
            return result.get('matches', [])
        else:
            logging.warning(f"LanguageTool API request failed with status code: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Error calling LanguageTool API: {str(e)}")
        return []

def process_languagetool_results(matches: List[Dict[str, Any]], focus_on_spoken: bool = True) -> List[str]:
    """
    Process the raw LanguageTool API results into a list of user-friendly issues
    When focus_on_spoken is True, filter for issues that would affect spoken English
    """
    issues = []
    
    # Categories that are particularly relevant for spoken English
    spoken_categories = {
        'GRAMMAR', 'TYPOS', 'CONFUSED_WORDS', 'COLLOCATIONS', 
        'REDUNDANCY', 'SEMANTICS', 'STYLE', 'GENDER_NEUTRALITY'
    }
    
    # Rule IDs to ignore for spoken English
    ignore_rules = {
        'PUNCTUATION', 'TYPOGRAPHY', 'WHITESPACE', 'COMMA', 'PERIOD', 
        'UPPERCASE_SENTENCE_START', 'EN_QUOTES', 'DASH_RULE'
    }
    
    for match in matches:
        rule_id = match.get('rule', {}).get('id', '')
        category_id = match.get('rule', {}).get('category', {}).get('id', '')
        
        # Skip punctuation and typography issues for spoken English
        if focus_on_spoken and (
            any(ignore_word in rule_id for ignore_word in ignore_rules) or
            category_id not in spoken_categories
        ):
            continue
            
        # Get the message and context
        message = match.get('message', '')
        context = match.get('context', {})
        text = context.get('text', '')
        offset = context.get('offset', 0)
        length = context.get('length', 0)
        
        # Get the problematic part of the text
        problem_text = text[offset:offset+length] if text else ""
        
        # Get suggestions
        replacements = [r.get('value', '') for r in match.get('replacements', [])][:3]  # Limit to top 3
        suggestion = f" -> {', '.join(replacements)}" if replacements else ""
        
        issues.append(f"{message}: '{problem_text}'{suggestion}")
    
    return issues

def check_grammar(text: str) -> Tuple[List[str], Optional[str]]:
    """
    Check the grammar of the text using a multi-tool approach:
    1. SpaCy for basic analysis
    2. TextBlob for additional spell checking
    3. PySpellChecker as primary spell checker
    4. LanguageTool API for advanced grammar checking
    
    Focused on vocabulary and structure errors relevant for spoken English
    Returns a list of issues and optionally a suggestion for correction
    """
    issues = []
    correction_suggestion = None

    if not text:
        return issues, correction_suggestion

    # Detect language
    lang = detect_language(text)
    languagetool_lang = 'en-US' if lang == 'en' else lang
    
    # Only process English text or unknown language text
    if lang not in ['en', 'unknown']:
        return [f"Non-English text detected (language: {lang})"], None
    
    # Step 1: Use PySpellChecker for spelling errors (only for English)
    pyspell_corrections = check_spelling_with_pyspellchecker(text)
    for original, suggestion in pyspell_corrections:
        issues.append(f"Possible vocabulary error: '{original}' -> '{suggestion}'")
    
    # Step 2: Use TextBlob as a backup for spell checking
    blob = TextBlob(text)
    
    # Find words that might be misspelled but weren't caught by PySpellChecker
    misspelled_words = []
    if nlp:
        for word in blob.words:
            # Skip words already found by PySpellChecker
            if any(word.lower() == misspelled[0].lower() for misspelled in pyspell_corrections):
                continue
                
            if word.isalpha() and len(word) > 2:  # Only check actual words
                if word.lower() not in nlp.vocab:
                    corrections = word.spellcheck()
                    if corrections and corrections[0][1] > 0.8:  # Only suggest if confidence is high
                        misspelled_words.append((word, corrections[0][0]))
        
        # Add TextBlob's suggestions
        for original, suggestion in misspelled_words:
            issues.append(f"Possible vocabulary error: '{original}' -> '{suggestion}'")
    
    # Step 3: Use LanguageTool for more advanced grammar checking
    languagetool_matches = check_with_languagetool(text, languagetool_lang)
    languagetool_issues = process_languagetool_results(languagetool_matches, focus_on_spoken=True)
    
    # Add LanguageTool issues to our list
    for issue in languagetool_issues:
        issues.append(issue)
    
    # Generate a corrected version of the text
    if issues:
        corrected_text = text
        
        # Apply PySpellChecker corrections
        for original, suggestion in pyspell_corrections:
            # Use word boundary pattern for more accurate replacement
            corrected_text = re.sub(r'\b' + re.escape(original) + r'\b', suggestion, corrected_text)
        
        # Apply TextBlob corrections
        for original, suggestion in misspelled_words:
            corrected_text = re.sub(r'\b' + re.escape(original) + r'\b', suggestion, corrected_text)
            
        # Apply LanguageTool corrections
        for match in languagetool_matches:
            # Only apply corrections that have suggestions
            if match.get('replacements'):
                context = match.get('context', {})
                text_fragment = context.get('text', '')
                offset = context.get('offset', 0)
                length = context.get('length', 0)
                
                if text_fragment and offset >= 0 and length > 0:
                    problem_text = text_fragment[offset:offset+length]
                    suggestion = match.get('replacements', [{}])[0].get('value', '')
                    
                    if problem_text and suggestion:
                        # Find the problem text in the original text to ensure we're replacing the right occurrence
                        corrected_text = corrected_text.replace(problem_text, suggestion, 1)
                        
        correction_suggestion = corrected_text
    
    return issues, correction_suggestion

def categorize_grammar_issues(issues: List[str]) -> Dict[str, List[str]]:
    """
    Categorize grammar issues into different types to better format the response
    """
    categories = {
        "vocabulary": [],
        "grammar": [],
        "style": [],
        "other": []
    }
    
    for issue in issues:
        if "vocabulary error" in issue.lower():
            categories["vocabulary"].append(issue)
        elif any(keyword in issue.lower() for keyword in ["grammar", "tense", "agreement", "verb"]):
            categories["grammar"].append(issue)
        elif any(keyword in issue.lower() for keyword in ["style", "redundant", "wordy"]):
            categories["style"].append(issue)
        else:
            categories["other"].append(issue)
    
    return categories

def needs_grammar_correction(text: str) -> bool:
    """
    Determine if a text needs grammar correction or language detection
    Returns True if the text is not in English or has significant vocabulary issues
    Ignores punctuation issues since this is a voice-powered app
    """
    if not text:
        return False
    
    # Detect language first - this is the key check
    language = detect_language(text)
    
    # If not English, we should respond with a language notice
    if language != 'en' and language != 'unknown':
        return True
    
    # Only process English text for grammar issues
    if language == 'en' or language == 'unknown':
        # Check for grammar issues
        issues, _ = check_grammar(text)
        
        # Check for substantive issues (not just punctuation)
        substantive_issues = [
            issue for issue in issues 
            if not any(p in issue.lower() for p in ['punctuation', 'comma', 'period', 'capitalization'])
        ]
        
        # Only suggest corrections if there are meaningful issues
        return len(substantive_issues) > 0
    
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
    
    # Detect language
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
    
    # Check grammar for English text
    issues, corrected_text = check_grammar(text)
    
    # Categorize the issues for better response formatting
    if issues:
        categorized_issues = categorize_grammar_issues(issues)
        
        return {
            "grammar_issues": {
                "language_detected": language,
                "issues": issues,
                "categorized_issues": categorized_issues,
                "needs_response": True
            },
            "corrected_text": corrected_text
        }
    
    return {"grammar_issues": None, "corrected_text": None}

def generate_grammar_response(state: models.EnhancedState, config: RunnableConfig, components: dict) -> Dict[str, Any]:
    """
    Generate a response about grammar issues for the user
    Focuses on vocabulary and sentence structure for spoken English
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
    categorized_issues = grammar_issues.get("categorized_issues", {})
    
    if language_detected != "en" and language_detected != "unknown":
        # Create a system message for non-English detection
        system_msg = """You are a helpful English tutor assistant. The user has sent a message in a language that's not English. 
        Respond ONLY in English that you currently only understand English. 
        If you can recognize the language, mention it in your response. 
        Be friendly and encouraging, as if you're talking to a language learner."""
        
        prompt = f"The user sent a message in '{language_detected}'. Respond politely IN ENGLISH ONLY that you only understand English, and encourage them to practice speaking in English."
    else:
        # Create a system message for grammar correction
        system_msg = """You are a friendly English tutor assistant that helps with spoken English.
        When users make vocabulary or grammar errors while speaking, you should politely point them out and suggest corrections.
        Be encouraging and conversational in your approach, explaining briefly how to improve.
        Focus only on errors that would hinder communication or understanding.
        Remember that you're correcting speech, not written text, so ignore minor issues that wouldn't matter in conversation.
        Always maintain a supportive tone to build the language learner's confidence.
        Keep your correction brief and natural, as if you're having a spoken conversation, then continue the main conversation flow."""
        
        # Format the grammar issues into a prompt
        # Use categorized issues for a more structured correction approach
        categorized_text = ""
        if categorized_issues:
            for category, category_issues in categorized_issues.items():
                if category_issues:
                    categorized_text += f"\n{category.capitalize()} issues:\n"
                    for issue in category_issues:
                        categorized_text += f"- {issue}\n"
        else:
            categorized_text = "\n".join([f"- {issue}" for issue in issues])
            
        prompt = f"""The user's spoken message had the following potential language issues:

{categorized_text}

Original text: "{state['question']}"
Suggested correction: "{corrected_text if corrected_text else state['question']}"

Provide a friendly, conversational response about these language issues as if you're having a spoken conversation.
Be encouraging and very briefly explain why the corrections help with better English speaking.
Then continue the conversation naturally, as if this correction was just a small part of your ongoing chat.
Keep your response brief, supportive, and focus on the most important improvements for spoken English.
Format your response naturally, not as a list of corrections - imagine this is a real-time spoken conversation."""

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