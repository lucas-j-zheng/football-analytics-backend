#!/usr/bin/env python3
"""
Add test game data for the test team
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:5001/api"

# Sample game data
sample_plays = [
    # Red zone plays
    {"play_id": 1, "down": 1, "distance": 10, "yard_line": 15, "formation": "Shotgun", "play_type": "Pass", "play_name": "Slant", "result_of_play": "Complete", "yards_gained": 8, "points_scored": 0},
    {"play_id": 2, "down": 2, "distance": 2, "yard_line": 7, "formation": "I-Form", "play_type": "Run", "play_name": "Dive", "result_of_play": "Rush", "yards_gained": 7, "points_scored": 6},
    {"play_id": 3, "down": 1, "distance": 10, "yard_line": 12, "formation": "Shotgun", "play_type": "Pass", "play_name": "Fade", "result_of_play": "Incomplete", "yards_gained": 0, "points_scored": 0},
    
    # Third down plays
    {"play_id": 4, "down": 3, "distance": 5, "yard_line": 35, "formation": "Shotgun", "play_type": "Pass", "play_name": "Quick Out", "result_of_play": "Complete", "yards_gained": 7, "points_scored": 0},
    {"play_id": 5, "down": 3, "distance": 8, "yard_line": 42, "formation": "Shotgun", "play_type": "Pass", "play_name": "Comeback", "result_of_play": "Complete", "yards_gained": 12, "points_scored": 0},
    {"play_id": 6, "down": 3, "distance": 3, "yard_line": 28, "formation": "I-Form", "play_type": "Run", "play_name": "Power", "result_of_play": "Rush", "yards_gained": 1, "points_scored": 0},
    
    # Running plays over 5 yards
    {"play_id": 7, "down": 1, "distance": 10, "yard_line": 25, "formation": "I-Form", "play_type": "Run", "play_name": "Outside Zone", "result_of_play": "Rush", "yards_gained": 8, "points_scored": 0},
    {"play_id": 8, "down": 2, "distance": 3, "yard_line": 33, "formation": "Singleback", "play_type": "Run", "play_name": "Draw", "result_of_play": "Rush", "yards_gained": 15, "points_scored": 0},
    {"play_id": 9, "down": 1, "distance": 10, "yard_line": 48, "formation": "Pistol", "play_type": "Run", "play_name": "Inside Zone", "result_of_play": "Rush", "yards_gained": 6, "points_scored": 0},
    
    # Big plays over 15 yards  
    {"play_id": 10, "down": 1, "distance": 10, "yard_line": 25, "formation": "Shotgun", "play_type": "Pass", "play_name": "Deep Post", "result_of_play": "Complete", "yards_gained": 22, "points_scored": 0},
    {"play_id": 11, "down": 2, "distance": 7, "yard_line": 47, "formation": "Shotgun", "play_type": "Pass", "play_name": "Go Route", "result_of_play": "Complete", "yards_gained": 35, "points_scored": 0},
    {"play_id": 12, "down": 1, "distance": 10, "yard_line": 18, "formation": "Singleback", "play_type": "Run", "play_name": "Sweep", "result_of_play": "Rush", "yards_gained": 18, "points_scored": 6},
    
    # More varied plays
    {"play_id": 13, "down": 1, "distance": 10, "yard_line": 50, "formation": "Shotgun", "play_type": "Pass", "play_name": "Screen", "result_of_play": "Complete", "yards_gained": 4, "points_scored": 0},
    {"play_id": 14, "down": 2, "distance": 6, "yard_line": 46, "formation": "I-Form", "play_type": "Run", "play_name": "Counter", "result_of_play": "Rush", "yards_gained": 3, "points_scored": 0},
    {"play_id": 15, "down": 3, "distance": 3, "yard_line": 43, "formation": "Shotgun", "play_type": "Pass", "play_name": "Hitch", "result_of_play": "Complete", "yards_gained": 5, "points_scored": 0},
    
    # Fourth down plays
    {"play_id": 16, "down": 4, "distance": 1, "yard_line": 38, "formation": "I-Form", "play_type": "Run", "play_name": "QB Sneak", "result_of_play": "Rush", "yards_gained": 2, "points_scored": 0},
    {"play_id": 17, "down": 4, "distance": 5, "yard_line": 33, "formation": "Shotgun", "play_type": "Pass", "play_name": "Curl", "result_of_play": "Incomplete", "yards_gained": 0, "points_scored": 0},
    
    # More shotgun formation plays
    {"play_id": 18, "down": 1, "distance": 10, "yard_line": 65, "formation": "Shotgun", "play_type": "Pass", "play_name": "Bubble Screen", "result_of_play": "Complete", "yards_gained": 6, "points_scored": 0},
    {"play_id": 19, "down": 2, "distance": 4, "yard_line": 59, "formation": "Shotgun", "play_type": "Pass", "play_name": "Quick Slant", "result_of_play": "Complete", "yards_gained": 8, "points_scored": 0},
    {"play_id": 20, "down": 1, "distance": 10, "yard_line": 51, "formation": "Shotgun", "play_type": "Run", "play_name": "Zone Read", "result_of_play": "Rush", "yards_gained": 4, "points_scored": 0}
]

def add_test_data():
    """Add test game data for the test team"""
    
    print("🏈 Adding test game data for LangChain testing...")
    print("=" * 50)
    
    try:
        # Login to get token
        print("\n🔑 Logging in...")
        login_response = requests.post(f"{BASE_URL}/auth/team/login", json={
            'email': 'test@eagles.com',
            'password': 'password123'
        })
        
        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.text}")
            return False
        
        token = login_response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        print("✅ Login successful")
        
        # Create game data
        game_data = {
            'week': 1,
            'opponent': 'Test Hawks',
            'location': 'Home',
            'analytics_focus_notes': 'Sample game data for LangChain testing - includes red zone, third down, and big plays'
        }
        
        print("\n📊 Creating test game...")
        game_response = requests.post(f"{BASE_URL}/games", json=game_data, headers=headers)
        
        if game_response.status_code != 201:
            print(f"❌ Game creation failed: {game_response.text}")
            return False
        
        game_id = game_response.json()['game']['id']
        print(f"✅ Test game created with ID: {game_id}")
        
        # Add play data
        print(f"\n🎯 Adding {len(sample_plays)} sample plays...")
        
        for i, play in enumerate(sample_plays, 1):
            play_data = {
                'game_id': game_id,
                'plays_data': [play]
            }
            
            play_response = requests.post(f"{BASE_URL}/games/{game_id}/plays", 
                                        json=play_data, headers=headers)
            
            if play_response.status_code == 201:
                print(f"  ✅ Play {i}/{len(sample_plays)} added")
            else:
                print(f"  ❌ Play {i} failed: {play_response.text}")
        
        print("\n" + "=" * 50)
        print("🎉 Test data successfully added!")
        print("\n📋 Sample plays include:")
        print("   • Red zone plays (yard lines 1-20)")
        print("   • Third down conversions")
        print("   • Running plays over 5 yards")
        print("   • Big plays over 15 yards")
        print("   • Shotgun formation passes")
        print("   • Various down and distance scenarios")
        
        print("\n🚀 Now you can test LangChain Assistant with queries like:")
        print("   • 'Show me red zone plays'")
        print("   • 'Third down conversions'")
        print("   • 'Running plays over 5 yards'")
        print("   • 'Big plays over 15 yards'")
        print("   • 'Shotgun formation passes'")
        print("=" * 50)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - make sure backend is running on port 5001")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    add_test_data()