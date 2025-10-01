#!/usr/bin/env python3
"""
Upload test game data CSV for the test team
"""
import requests
import os

BASE_URL = "http://localhost:5001/api"

def upload_test_data():
    """Upload test game data using CSV file"""
    
    print("🏈 Uploading test game data for LangChain testing...")
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
        headers = {'Authorization': f'Bearer {token}'}
        print("✅ Login successful")
        
        # Upload CSV file
        csv_file_path = "test_game_data.csv"
        if not os.path.exists(csv_file_path):
            print(f"❌ CSV file not found: {csv_file_path}")
            return False
        
        print(f"\n📤 Uploading CSV file: {csv_file_path}")
        
        # Prepare form data
        form_data = {
            'week': '1',
            'opponent': 'Test Hawks',
            'location': 'Home',
            'analytics_focus_notes': 'Sample game data for LangChain testing - includes red zone, third down, and big plays'
        }
        
        with open(csv_file_path, 'rb') as f:
            files = {'csv_file': ('test_game_data.csv', f, 'text/csv')}
            
            response = requests.post(f"{BASE_URL}/games", 
                                   data=form_data, 
                                   files=files, 
                                   headers=headers)
        
        if response.status_code == 201:
            game_data = response.json()
            game_id = game_data['game']['id']
            play_count = len(game_data.get('plays_added', []))
            
            print("✅ Game data uploaded successfully!")
            print(f"   Game ID: {game_id}")
            print(f"   Plays added: {play_count}")
            
            print("\n" + "=" * 50)
            print("🎉 Test data successfully uploaded!")
            print("\n📋 The uploaded data includes:")
            print("   • Red zone plays (yard lines 1-20)")
            print("   • Third down conversions")  
            print("   • Running plays over 5 yards")
            print("   • Big plays over 15+ yards")
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
        else:
            print(f"❌ Upload failed: {response.text}")
            return False
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - make sure backend is running on port 5001")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    upload_test_data()