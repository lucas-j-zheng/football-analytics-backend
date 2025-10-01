#!/usr/bin/env python3
"""
Create test games and play data directly in the database for the test team
"""
from app import app, db, Team, Game, PlayData
from datetime import datetime, timedelta

def create_test_games():
    """Create test games with play data for the test team"""
    
    with app.app_context():
        print("🏈 Creating test games for LangChain testing...")
        print("=" * 50)
        
        # Find the test team
        test_team = Team.query.filter_by(email='test@eagles.com').first()
        if not test_team:
            print("❌ Test team not found. Please run setup_test_data.py first.")
            return False
        
        print(f"✅ Found test team: {test_team.team_name} (ID: {test_team.id})")
        
        # Check if games already exist
        existing_games = Game.query.filter_by(team_id=test_team.id).count()
        if existing_games > 0:
            print(f"📊 Test team already has {existing_games} games. Clearing existing data...")
            # Delete existing plays and games
            # First get game IDs for this team
            game_ids = [game.id for game in Game.query.filter_by(team_id=test_team.id).all()]
            # Delete plays for those games
            if game_ids:
                PlayData.query.filter(PlayData.game_id.in_(game_ids)).delete(synchronize_session=False)
            # Delete games
            Game.query.filter_by(team_id=test_team.id).delete()
            db.session.commit()
        
        # Create test game
        game = Game(
            team_id=test_team.id,
            week=1,
            opponent='Test Hawks',
            location='Home',
            submission_timestamp=datetime.now(),
            analytics_focus_notes='Sample game data for LangChain testing - includes red zone, third down, and big plays'
        )
        
        db.session.add(game)
        db.session.commit()
        
        game_id = game.id
        print(f"✅ Created test game: Week {game.week} vs {game.opponent} (ID: {game_id})")
        
        # Sample play data optimized for LangChain testing
        plays_data = [
            # Red zone plays (yard lines 1-20)
            {"play_id": 1, "down": 1, "distance": 10, "yard_line": 15, "formation": "Shotgun", "play_type": "Pass", "play_name": "Slant", "result_of_play": "Complete", "yards_gained": 8, "points_scored": 0},
            {"play_id": 2, "down": 2, "distance": 2, "yard_line": 7, "formation": "I Formation", "play_type": "Run", "play_name": "Dive", "result_of_play": "Rush", "yards_gained": 7, "points_scored": 6},
            {"play_id": 3, "down": 1, "distance": 10, "yard_line": 12, "formation": "Shotgun", "play_type": "Pass", "play_name": "Fade", "result_of_play": "Incomplete", "yards_gained": 0, "points_scored": 0},
            {"play_id": 4, "down": 1, "distance": 10, "yard_line": 18, "formation": "Singleback", "play_type": "Run", "play_name": "Sweep", "result_of_play": "Rush", "yards_gained": 18, "points_scored": 6},
            {"play_id": 5, "down": 2, "distance": 5, "yard_line": 10, "formation": "I Formation", "play_type": "Run", "play_name": "Power", "result_of_play": "Rush", "yards_gained": 10, "points_scored": 6},
            
            # Third down plays  
            {"play_id": 6, "down": 3, "distance": 5, "yard_line": 35, "formation": "Shotgun", "play_type": "Pass", "play_name": "Quick Out", "result_of_play": "Complete", "yards_gained": 7, "points_scored": 0},
            {"play_id": 7, "down": 3, "distance": 8, "yard_line": 42, "formation": "Shotgun", "play_type": "Pass", "play_name": "Comeback", "result_of_play": "Complete", "yards_gained": 12, "points_scored": 0},
            {"play_id": 8, "down": 3, "distance": 3, "yard_line": 28, "formation": "I Formation", "play_type": "Run", "play_name": "Power", "result_of_play": "Rush", "yards_gained": 1, "points_scored": 0},
            {"play_id": 9, "down": 3, "distance": 7, "yard_line": 45, "formation": "Shotgun", "play_type": "Pass", "play_name": "Slant", "result_of_play": "Complete", "yards_gained": 9, "points_scored": 0},
            {"play_id": 10, "down": 3, "distance": 4, "yard_line": 26, "formation": "Shotgun", "play_type": "Pass", "play_name": "Quick Slant", "result_of_play": "Complete", "yards_gained": 8, "points_scored": 0},
            
            # Running plays over 5 yards
            {"play_id": 11, "down": 1, "distance": 10, "yard_line": 25, "formation": "I Formation", "play_type": "Run", "play_name": "Outside Zone", "result_of_play": "Rush", "yards_gained": 8, "points_scored": 0},
            {"play_id": 12, "down": 2, "distance": 3, "yard_line": 33, "formation": "Singleback", "play_type": "Run", "play_name": "Draw", "result_of_play": "Rush", "yards_gained": 15, "points_scored": 0},
            {"play_id": 13, "down": 1, "distance": 10, "yard_line": 48, "formation": "Pistol", "play_type": "Run", "play_name": "Inside Zone", "result_of_play": "Rush", "yards_gained": 6, "points_scored": 0},
            {"play_id": 14, "down": 2, "distance": 6, "yard_line": 46, "formation": "I Formation", "play_type": "Run", "play_name": "Counter", "result_of_play": "Rush", "yards_gained": 9, "points_scored": 0},
            {"play_id": 15, "down": 1, "distance": 10, "yard_line": 30, "formation": "Singleback", "play_type": "Run", "play_name": "Toss", "result_of_play": "Rush", "yards_gained": 12, "points_scored": 0},
            
            # Big plays over 15 yards
            {"play_id": 16, "down": 1, "distance": 10, "yard_line": 25, "formation": "Shotgun", "play_type": "Pass", "play_name": "Deep Post", "result_of_play": "Complete", "yards_gained": 22, "points_scored": 0},
            {"play_id": 17, "down": 2, "distance": 7, "yard_line": 47, "formation": "Shotgun", "play_type": "Pass", "play_name": "Go Route", "result_of_play": "Complete", "yards_gained": 35, "points_scored": 0},
            {"play_id": 18, "down": 1, "distance": 10, "yard_line": 40, "formation": "Shotgun", "play_type": "Pass", "play_name": "Deep Comeback", "result_of_play": "Complete", "yards_gained": 18, "points_scored": 0},
            {"play_id": 19, "down": 2, "distance": 8, "yard_line": 42, "formation": "Shotgun", "play_type": "Pass", "play_name": "Out Route", "result_of_play": "Complete", "yards_gained": 25, "points_scored": 0},
            
            # Shotgun formation passes
            {"play_id": 20, "down": 1, "distance": 10, "yard_line": 50, "formation": "Shotgun", "play_type": "Pass", "play_name": "Screen", "result_of_play": "Complete", "yards_gained": 4, "points_scored": 0},
            {"play_id": 21, "down": 1, "distance": 10, "yard_line": 65, "formation": "Shotgun", "play_type": "Pass", "play_name": "Bubble Screen", "result_of_play": "Complete", "yards_gained": 6, "points_scored": 0},
            {"play_id": 22, "down": 2, "distance": 4, "yard_line": 59, "formation": "Shotgun", "play_type": "Pass", "play_name": "Quick Slant", "result_of_play": "Complete", "yards_gained": 8, "points_scored": 0},
            {"play_id": 23, "down": 1, "distance": 10, "yard_line": 51, "formation": "Shotgun", "play_type": "Run", "play_name": "Zone Read", "result_of_play": "Rush", "yards_gained": 4, "points_scored": 0},
            {"play_id": 24, "down": 2, "distance": 6, "yard_line": 55, "formation": "Shotgun", "play_type": "Pass", "play_name": "Hitch", "result_of_play": "Complete", "yards_gained": 7, "points_scored": 0},
            
            # Fourth down plays
            {"play_id": 25, "down": 4, "distance": 1, "yard_line": 38, "formation": "I Formation", "play_type": "Run", "play_name": "QB Sneak", "result_of_play": "Rush", "yards_gained": 2, "points_scored": 0},
            {"play_id": 26, "down": 4, "distance": 5, "yard_line": 33, "formation": "Shotgun", "play_type": "Pass", "play_name": "Curl", "result_of_play": "Incomplete", "yards_gained": 0, "points_scored": 0},
            
            # More variety for comprehensive testing
            {"play_id": 27, "down": 1, "distance": 10, "yard_line": 75, "formation": "I Formation", "play_type": "Run", "play_name": "Fullback Dive", "result_of_play": "Rush", "yards_gained": 3, "points_scored": 0},
            {"play_id": 28, "down": 2, "distance": 7, "yard_line": 72, "formation": "Shotgun", "play_type": "Pass", "play_name": "Cross", "result_of_play": "Complete", "yards_gained": 14, "points_scored": 0},
            {"play_id": 29, "down": 1, "distance": 10, "yard_line": 58, "formation": "Gun Trips", "play_type": "Pass", "play_name": "Fade", "result_of_play": "Incomplete", "yards_gained": 0, "points_scored": 0},
            {"play_id": 30, "down": 2, "distance": 10, "yard_line": 58, "formation": "Shotgun", "play_type": "Pass", "play_name": "Checkdown", "result_of_play": "Complete", "yards_gained": 5, "points_scored": 0}
        ]
        
        # Add play data to database
        play_count = 0
        for play_data in plays_data:
            play = PlayData(
                game_id=game_id,
                play_id=play_data['play_id'],
                down=play_data['down'],
                distance=play_data['distance'],
                yard_line=play_data['yard_line'],
                formation=play_data['formation'],
                play_type=play_data['play_type'],
                play_name=play_data['play_name'],
                result_of_play=play_data['result_of_play'],
                yards_gained=play_data['yards_gained'],
                points_scored=play_data['points_scored'],
                unit="O"  # Offensive plays
            )
            db.session.add(play)
            play_count += 1
        
        db.session.commit()
        
        print(f"✅ Added {play_count} plays to the game")
        
        print("\n" + "=" * 50)
        print("🎉 Test data successfully created!")
        print("\n📋 The test game includes:")
        print(f"   • {len([p for p in plays_data if p['yard_line'] <= 20])} Red zone plays (yard lines 1-20)")
        print(f"   • {len([p for p in plays_data if p['down'] == 3])} Third down plays")
        print(f"   • {len([p for p in plays_data if p['play_type'] == 'Run' and p['yards_gained'] > 5])} Running plays over 5 yards")
        print(f"   • {len([p for p in plays_data if p['yards_gained'] > 15])} Big plays over 15 yards")
        print(f"   • {len([p for p in plays_data if 'Shotgun' in p['formation']])} Shotgun formation plays")
        print(f"   • {len([p for p in plays_data if p['down'] == 4])} Fourth down plays")
        
        print("\n🚀 Now you can test LangChain Assistant with queries like:")
        print("   • 'Show me red zone plays'")
        print("   • 'Third down conversions'")
        print("   • 'Running plays over 5 yards'")
        print("   • 'Big plays over 15 yards'")
        print("   • 'Shotgun formation passes'")
        print("   • 'Fourth down attempts'")
        print("=" * 50)
        
        return True

if __name__ == "__main__":
    create_test_games()