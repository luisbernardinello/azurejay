import pytest
from langchain_core.messages import HumanMessage

from src.agent import grammar
from src.agent.models import EnhancedState

def test_grammar_detection_and_correction():
    """Test the grammar detection and correction functionality"""
    test_cases = [
        {
            "input": "I has went to the store yesterday",
            "expected_issues": ["verb tense", "has went"],
            "language": "en"
        },
        {
            "input": "She dont like apples",
            "expected_issues": ["doesn't", "don't"],
            "language": "en"
        },
        {
            "input": "I very tired today",
            "expected_issues": ["missing verb", "am"],
            "language": "en"
        },
        {
            "input": "Hola, como estas?",
            "expected_issues": ["Non-English"],
            "language": "es"
        }
    ]
    
    for test_case in test_cases:
        state = EnhancedState(
            messages=[HumanMessage(content=test_case["input"])],
            question=test_case["input"],
            answer="",
            memory={},
            context=[],
            search_needed=False,
            grammar_issues=None,
            corrected_text=None
        )
        

        result = grammar.process_grammar(state)
        
        assert result["grammar_issues"]["language_detected"] == test_case["language"]
        
        assert result["grammar_issues"]["needs_response"] == True
        
        issues_text = " ".join(result["grammar_issues"]["issues"]).lower()
        for expected_issue in test_case["expected_issues"]:
            assert expected_issue.lower() in issues_text
        
        if test_case["language"] == "en":
            assert result["corrected_text"] is not None

def test_categorize_grammar_issues():
    """Test the categorization of grammar issues"""
    issues = [
        "Possible vocabulary error: 'wich' -> 'which'",
        "Subject-verb agreement error: 'they was' -> 'they were'",
        "Redundant phrase: 'return back' -> 'return'",
        "Capitalization issue: 'i am' -> 'I am'"
    ]
    
    categorized = grammar.categorize_grammar_issues(issues)
    
    assert len(categorized["vocabulary"]) == 1
    assert "wich" in categorized["vocabulary"][0]
    
    assert len(categorized["grammar"]) == 1
    assert "they was" in categorized["grammar"][0]
    
    assert len(categorized["style"]) == 1
    assert "redundant" in categorized["style"][0].lower()

def test_languagetool_integration():
    """Test the LanguageTool API integration"""
    text = "I doesnt beleive in ghosts"
    
    matches = grammar.check_with_languagetool(text)
    
    issues = grammar.process_languagetool_results(matches)
    

    combined_issues = " ".join(issues).lower()
    assert "doesn't" in combined_issues or "does not" in combined_issues
    assert "believe" in combined_issues