#!/usr/bin/env python3
"""
Setup test data for LangChain Assistant
"""
import requests
import json

BASE_URL = "http://localhost:5001/api"

def setup_test_data():
    """Create test team and upload sample data"""
    
    # Test team data
    test_team = {
        'team_name': 'Test Eagles',
        'email': 'test@eagles.com',
        'password': 'password123',
        'school_name': 'Test University',
        'coach_name': 'Coach Johnson'
    }
    
    print("🏈 Setting up test data for LangChain Assistant")
    print("=" * 50)
    
    try:
        # Create test team
        print("\n📊 Creating test team...")
        response = requests.post(f"{BASE_URL}/auth/team/register", json=test_team)
        if response.status_code == 201:
            print("✅ Test team created successfully")
        elif response.status_code == 409:
            print("📝 Test team already exists")
        else:
            print(f"❌ Failed to create team: {response.text}")
            return False
        
        # Login to get token
        print("\n🔑 Logging in...")
        login_response = requests.post(f"{BASE_URL}/auth/team/login", json={
            'email': test_team['email'],
            'password': test_team['password']
        })
        
        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.text}")
            return False
        
        token = login_response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        
        print("✅ Login successful")
        
        # Check if we have games
        games_response = requests.get(f"{BASE_URL}/games", headers=headers)
        if games_response.status_code == 200:
            games = games_response.json()['games']
            print(f"\n📋 Found {len(games)} existing games")
            
            if len(games) == 0:
                print("📤 No games found. You'll need to upload game data through the UI first.")
                print("   1. Go to http://localhost:3001")
                print("   2. Login with: test@eagles.com / password123")
                print("   3. Upload game data using the sample CSV file")
        
        # Test LangChain endpoints
        print("\n🤖 Testing LangChain endpoints...")
        
        # Test status
        status_response = requests.get(f"{BASE_URL}/langchain/status", headers=headers)
        if status_response.status_code == 200:
            print("✅ LangChain status endpoint working")
        else:
            print(f"❌ LangChain status failed: {status_response.text}")
        
        # Test translation
        translate_response = requests.post(f"{BASE_URL}/langchain/translate", 
                                         json={'query': 'show me red zone plays'}, 
                                         headers=headers)
        if translate_response.status_code == 200:
            print("✅ LangChain translate endpoint working")
        else:
            print(f"❌ LangChain translate failed: {translate_response.text}")
        
        print("\n" + "=" * 50)
        print("🎉 Test setup completed!")
        print("\n📋 To test LangChain Assistant:")
        print("   1. Go to: http://localhost:3001")
        print("   2. Login with: test@eagles.com / password123")
        print("   3. Click the '🚀 LangChain AI' button")
        print("   4. Try queries like:")
        print("      • 'Show me red zone plays'")
        print("      • 'Third down conversions'")
        print("      • 'Running plays over 5 yards'")
        print("=" * 50)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - make sure backend is running on port 5001")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    setup_test_data()