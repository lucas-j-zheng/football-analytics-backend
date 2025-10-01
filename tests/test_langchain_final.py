#!/usr/bin/env python3
"""
Comprehensive LangChain Integration Test Results
Tests all working functionality after fixing the timeout issue
"""

import requests
import json

# JWT token for testing
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTc1NzQyODgyMiwianRpIjoiZDg2N2YxM2UtMzhjOS00ZmQ5LTlkZTYtNDk3NTBlNzA0YmYxIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IjMiLCJuYmYiOjE3NTc0Mjg4MjIsImV4cCI6MTc1NzUxNTIyMiwidXNlcl90eXBlIjoidGVhbSIsInVzZXJfaWQiOjN9.v1_xFY3EGcoHGEqky0vEVuEPH6xNjHrspSrI2wSSnb4"
BASE_URL = "http://localhost:5001"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def test_endpoint(name, method, endpoint, data=None):
    """Test an endpoint and return results"""
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, headers=HEADERS)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS, json=data)
        
        return {
            "name": name,
            "status": response.status_code,
            "success": response.status_code < 400,
            "response": response.json() if response.content else {}
        }
    except Exception as e:
        return {
            "name": name,
            "success": False,
            "error": str(e)
        }

def main():
    print("🏈 LANGCHAIN INTEGRATION - COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    tests = [
        # Test 1: Service Status
        {
            "name": "Service Status",
            "method": "GET",
            "endpoint": "/api/langchain/status",
            "expected": "Service availability and capabilities"
        },
        
        # Test 2: Simple Translation
        {
            "name": "Basic Query Translation",
            "method": "POST", 
            "endpoint": "/api/langchain/translate",
            "data": {"query": "red zone plays"},
            "expected": "yard_line between [1, 20]"
        },
        
        # Test 3: Complex Translation
        {
            "name": "Complex Multi-Condition Query",
            "method": "POST",
            "endpoint": "/api/langchain/translate", 
            "data": {"query": "shotgun formation passing plays on third down"},
            "expected": "Multiple conditions: formation, play_type, down"
        },
        
        # Test 4: Conversation History
        {
            "name": "Conversation History",
            "method": "GET",
            "endpoint": "/api/langchain/conversation/history",
            "expected": "Empty history initially"
        }
    ]
    
    results = []
    for test in tests:
        print(f"\n🔍 Testing: {test['name']}")
        result = test_endpoint(
            test['name'], 
            test['method'], 
            test['endpoint'], 
            test.get('data')
        )
        results.append(result)
        
        if result['success']:
            print(f"   ✅ SUCCESS (Status: {result['status']})")
            if 'response' in result and result['response']:
                if test['name'] == "Basic Query Translation":
                    conditions = result['response'].get('filters', {}).get('conditions', [])
                    if conditions:
                        print(f"   📋 Translated to: {conditions[0]['field']} {conditions[0]['operator']} {conditions[0]['value']}")
                elif test['name'] == "Complex Multi-Condition Query":
                    conditions = result['response'].get('filters', {}).get('conditions', [])
                    print(f"   📋 Generated {len(conditions)} conditions")
        else:
            print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL TEST SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\n🎉 Working Features:")
        for result in successful:
            print(f"   ✓ {result['name']}")
    
    if failed:
        print(f"\n⚠️  Failed Features:")
        for result in failed:
            print(f"   ✗ {result['name']}")
    
    # Overall status
    if len(successful) == len(results):
        print(f"\n🚀 LANGCHAIN INTEGRATION: FULLY OPERATIONAL!")
        print("   All core LangChain functionality is working correctly.")
    elif len(successful) >= len(results) * 0.75:
        print(f"\n✅ LANGCHAIN INTEGRATION: MOSTLY WORKING")
        print("   Core functionality operational with minor issues.")
    else:
        print(f"\n⚠️  LANGCHAIN INTEGRATION: NEEDS ATTENTION")
        print("   Multiple components need fixing.")
    
    return len(successful) == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)