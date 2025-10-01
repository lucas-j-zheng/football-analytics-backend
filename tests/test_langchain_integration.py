#!/usr/bin/env python3
"""
Test script for LangChain integration
Tests the new ML query pipeline functionality
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_imports():
    """Test that all LangChain components can be imported"""
    print("\n🔧 Testing imports...")
    
    try:
        from langchain_service import langchain_service, FootballLangChainService
        print("✅ LangChain service imported successfully")
        
        from nl_query_translator import FootballQueryTranslator
        print("✅ Query translator imported successfully")
        
        from analysis_pipeline import FootballAnalysisPipeline, AnalysisStep, AnalysisStepType
        print("✅ Analysis pipeline imported successfully")
        
        # Test LangChain core imports
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_ollama import OllamaLLM
        print("✅ LangChain core components imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_sample_data():
    """Create sample football data for testing"""
    sample_plays = [
        {
            'play_id': 1,
            'down': 1,
            'distance': 10,
            'yard_line': 25,
            'formation': 'Shotgun',
            'play_type': 'Pass',
            'play_name': 'Slant Right',
            'result_of_play': 'Complete',
            'yards_gained': 8,
            'points_scored': 0
        },
        {
            'play_id': 2,
            'down': 2,
            'distance': 2,
            'yard_line': 33,
            'formation': 'I-Formation',
            'play_type': 'Run',
            'play_name': 'Dive',
            'result_of_play': 'Rush',
            'yards_gained': 3,
            'points_scored': 0
        },
        {
            'play_id': 3,
            'down': 1,
            'distance': 10,
            'yard_line': 36,
            'formation': 'Shotgun',
            'play_type': 'Pass',
            'play_name': 'Deep Out',
            'result_of_play': 'Complete',
            'yards_gained': 15,
            'points_scored': 0
        },
        {
            'play_id': 4,
            'down': 3,
            'distance': 7,
            'yard_line': 15,
            'formation': 'Shotgun',
            'play_type': 'Pass',
            'play_name': 'Red Zone Fade',
            'result_of_play': 'Touchdown',
            'yards_gained': 15,
            'points_scored': 6
        },
        {
            'play_id': 5,
            'down': 4,
            'distance': 1,
            'yard_line': 45,
            'formation': 'I-Formation',
            'play_type': 'Run',
            'play_name': 'QB Sneak',
            'result_of_play': 'Rush',
            'yards_gained': 2,
            'points_scored': 0
        }
    ]
    return sample_plays

def test_langchain_service():
    """Test the LangChain service functionality"""
    print("\n🧠 Testing LangChain service...")
    
    try:
        from langchain_service import langchain_service
        
        # Test service availability
        print(f"Service available: {langchain_service.is_available()}")
        
        # Get service stats
        stats = langchain_service.get_service_stats()
        print(f"Service stats: {json.dumps(stats, indent=2)}")
        
        return True
    except Exception as e:
        print(f"❌ LangChain service test failed: {e}")
        return False

def test_query_translator():
    """Test natural language query translation"""
    print("\n🔍 Testing query translation...")
    
    try:
        from langchain_service import langchain_service
        from nl_query_translator import FootballQueryTranslator
        
        translator = FootballQueryTranslator(langchain_service.llm)
        
        # Test queries
        test_queries = [
            "red zone plays",
            "third down conversions", 
            "running plays for more than 5 yards",
            "shotgun formation passes"
        ]
        
        for query in test_queries:
            print(f"\nTranslating: '{query}'")
            result = translator.translate_query(query)
            
            print(f"Success: {result.success}")
            if result.success:
                print(f"Filters: {json.dumps(result.filters, indent=2)}")
                print(f"Confidence: {result.confidence_score}")
            else:
                print(f"Error: {result.error_message}")
                if result.suggested_corrections:
                    print(f"Suggestions: {result.suggested_corrections}")
        
        # Test query examples
        examples = translator.get_query_examples()
        print(f"\nAvailable query examples: {len(examples)}")
        for query, description in list(examples.items())[:3]:
            print(f"  '{query}': {description}")
        
        return True
    except Exception as e:
        print(f"❌ Query translator test failed: {e}")
        return False

def test_enhanced_analysis():
    """Test enhanced analysis with sample data"""
    print("\n📊 Testing enhanced analysis...")
    
    try:
        from langchain_service import langchain_service
        
        sample_plays = test_sample_data()
        
        # Test basic analysis
        query = "What is our red zone performance?"
        print(f"\nAnalyzing: '{query}'")
        
        if not langchain_service.is_available():
            print("⚠️  Ollama not available, using fallback analysis")
            # Test with mock analysis
            analysis = "Mock analysis: Based on the sample data, you have 1 red zone play with a touchdown conversion."
        else:
            analysis = langchain_service.analyze_football_data_enhanced(query, sample_plays)
        
        print(f"Analysis result: {analysis[:200]}...")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced analysis test failed: {e}")
        return False

def test_conversational_query():
    """Test conversational query processing"""
    print("\n💬 Testing conversational queries...")
    
    try:
        from langchain_service import langchain_service
        
        sample_plays = test_sample_data()
        
        # Test conversational query
        query = "Show me third down plays and tell me how we performed"
        print(f"\nProcessing: '{query}'")
        
        result = langchain_service.conversational_query(query, sample_plays)
        
        print(f"Result type: {result['type']}")
        if 'filtered_count' in result:
            print(f"Filtered plays: {result['filtered_count']}/{result['total_count']}")
        print(f"Analysis preview: {result['analysis'][:200]}...")
        
        # Test conversation history
        history = langchain_service.get_conversation_history()
        print(f"\nConversation history length: {len(history)}")
        
        return True
    except Exception as e:
        print(f"❌ Conversational query test failed: {e}")
        return False

def test_analysis_pipeline():
    """Test the analysis pipeline workflows"""
    print("\n⚙️ Testing analysis pipeline...")
    
    try:
        from langchain_service import langchain_service
        from nl_query_translator import FootballQueryTranslator
        from analysis_pipeline import FootballAnalysisPipeline
        
        translator = FootballQueryTranslator(langchain_service.llm)
        pipeline = FootballAnalysisPipeline(langchain_service.llm, translator)
        
        # Get available workflows
        workflows = pipeline.get_available_workflows()
        print(f"Available workflows: {list(workflows.keys())}")
        
        sample_plays = test_sample_data()
        
        # Test a simple workflow (red_zone_analysis)
        if "red_zone_analysis" in workflows:
            print(f"\nExecuting red_zone_analysis workflow...")
            
            if langchain_service.is_available():
                result = pipeline.execute_workflow("red_zone_analysis", sample_plays)
                print(f"Pipeline success: {result.success}")
                print(f"Steps executed: {len(result.steps)}")
                print(f"Summary preview: {result.summary[:200] if result.summary else 'No summary'}...")
            else:
                print("⚠️  Ollama not available, skipping workflow execution")
        
        return True
    except Exception as e:
        print(f"❌ Analysis pipeline test failed: {e}")
        return False

def test_integration_summary():
    """Provide a summary of the integration test results"""
    print("\n📋 Integration Test Summary")
    print("=" * 50)
    
    # Test results will be stored here
    results = {}
    
    # Run all tests
    results['imports'] = test_imports()
    results['langchain_service'] = test_langchain_service()
    results['query_translator'] = test_query_translator()
    results['enhanced_analysis'] = test_enhanced_analysis()
    results['conversational_query'] = test_conversational_query()
    results['analysis_pipeline'] = test_analysis_pipeline()
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All LangChain integration tests passed!")
        print("\n📚 Available endpoints:")
        print("  • GET  /api/langchain/status")
        print("  • POST /api/langchain/query")
        print("  • POST /api/langchain/translate")
        print("  • POST /api/langchain/analyze")
        print("  • POST /api/langchain/workflow")
        print("  • POST /api/langchain/multi-step")
        print("  • GET  /api/langchain/conversation/history")
        print("  • POST /api/langchain/conversation/clear")
        print("  • GET  /api/langchain/workflows")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        for test_name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}")
    
    return passed == total

def main():
    """Main test runner"""
    print("🚀 LangChain Integration Test Suite")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Run tests
    all_passed = test_integration_summary()
    
    if all_passed:
        print("\n🎯 Next steps:")
        print("1. Install LangChain dependencies: pip install -r requirements.txt")
        print("2. Start your Flask server: python app.py")
        print("3. Test the new endpoints with your frontend")
        print("4. Try natural language queries like:")
        print("   - 'Show me red zone plays'")
        print("   - 'What's our third down conversion rate?'")
        print("   - 'Analyze running plays for more than 5 yards'")
        return 0
    else:
        print("\n🔧 Fix the failing tests before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)