#!/usr/bin/env python3

"""
Simple test to verify LangChain natural language query translation works correctly
"""

import sys
sys.path.append('.')

from nl_query_translator import FootballQueryTranslator
from langchain_ollama import OllamaLLM

def test_natural_language_translation():
    """Test the natural language translation functionality"""
    print("🏈 Testing Football Query Translation...")
    
    try:
        # Initialize LLM
        llm = OllamaLLM(base_url="http://localhost:11434", model="llama3.2:1b", temperature=0.3)
        
        # Initialize translator
        translator = FootballQueryTranslator(llm)
        
        # Test queries
        test_queries = [
            "show me red zone plays",
            "third down conversions",
            "running plays for more than 5 yards",
            "shotgun formation passes",
            "big plays over 15 yards"
        ]
        
        print(f"\n✅ Testing {len(test_queries)} queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: '{query}'")
            result = translator.translate_query(query)
            
            if result.success:
                print(f"  ✅ SUCCESS - Confidence: {result.confidence_score:.2f}")
                print(f"  📋 Conditions: {len(result.filters['conditions'])} filter(s)")
                for condition in result.filters['conditions']:
                    print(f"    - {condition['field']} {condition['operator']} {condition['value']}")
            else:
                print(f"  ❌ FAILED: {result.error_message}")
            print()
        
        # Test difficulty analysis
        print("🎯 Testing difficulty analysis...")
        for query in test_queries[:2]:
            analysis = translator.analyze_query_difficulty(query)
            print(f"'{query}' -> Difficulty: {analysis['difficulty']} (score: {analysis['complexity_score']:.2f})")
        
        print("\n🎉 Natural language translation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_natural_language_translation()
    sys.exit(0 if success else 1)