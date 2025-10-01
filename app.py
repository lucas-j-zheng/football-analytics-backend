from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_socketio import SocketIO
from datetime import timedelta, datetime
import os
import csv
import io
import re
import logging
from dotenv import load_dotenv
from app.utils.jwt_helper import get_current_user
from app.services.ai_local import local_ai
from app.services.langchain_service import langchain_service
from app.services.nl_query_translator import FootballQueryTranslator
from app.services.analysis_pipeline import FootballAnalysisPipeline
from app.api.collaboration import CollaborationService
from app.api.reporting import init_reporting, report_generator

load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("SECRET_KEY environment variable must be set")

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///football_analytics.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
if not app.config['JWT_SECRET_KEY']:
    raise ValueError("JWT_SECRET_KEY environment variable must be set")

app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
ma = Marshmallow(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# CORS Configuration - allow frontend URL from environment
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3001')
allowed_origins = [frontend_url, "http://localhost:3001", "http://localhost:3000"]
CORS(app, origins=allowed_origins)
socketio = SocketIO(app, cors_allowed_origins=allowed_origins)

# Initialize collaboration service
collaboration_service = CollaborationService(socketio, db)
collaboration_service.init_events()

# Initialize reporting service
init_reporting(db)

# Initialize FootballViz API
from app.api.footballviz_api import init_footballviz_api
footballviz_api = init_footballviz_api(app, db, socketio)

# Initialize LangChain components
query_translator = FootballQueryTranslator(langchain_service.llm)
analysis_pipeline = FootballAnalysisPipeline(langchain_service.llm, query_translator)

# Models
class Team(db.Model):
    __tablename__ = 'teams'
    
    id = db.Column(db.Integer, primary_key=True)
    team_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    # Relationships
    games = db.relationship('Game', backref='team', lazy=True)

class Consultant(db.Model):
    __tablename__ = 'consultants'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Game(db.Model):
    __tablename__ = 'games'
    
    id = db.Column(db.Integer, primary_key=True)
    team_id = db.Column(db.Integer, db.ForeignKey('teams.id'), nullable=False)
    week = db.Column(db.Integer, nullable=False)
    opponent = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(10), nullable=False)  # 'Home' or 'Away'
    analytics_focus_notes = db.Column(db.Text)
    csv_file_path = db.Column(db.String(255))
    submission_timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    # Relationships
    play_data = db.relationship('PlayData', backref='game', lazy=True, cascade='all, delete-orphan')
    visualizations = db.relationship('Visualization', backref='game', lazy=True)

class PlayData(db.Model):
    __tablename__ = 'play_data'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('games.id'), nullable=False)
    play_id = db.Column(db.Integer, nullable=False)
    down = db.Column(db.Integer, nullable=True)  # Allow null for special teams
    distance = db.Column(db.Integer, nullable=True)  # Allow null for special teams
    yard_line = db.Column(db.Integer, nullable=False)
    formation = db.Column(db.String(100), nullable=False)
    play_type = db.Column(db.String(50), nullable=False)
    play_name = db.Column(db.String(100), nullable=False)
    result_of_play = db.Column(db.String(100), nullable=False)
    yards_gained = db.Column(db.Integer, default=0)
    points_scored = db.Column(db.Integer, default=0)
    unit = db.Column(db.String(20), nullable=False)  # O, D, ST (offense, defense, special teams)
    
    # Additional optional columns for future expansion
    quarter = db.Column(db.Integer)
    time_remaining = db.Column(db.String(10))
    score_home = db.Column(db.Integer)
    score_away = db.Column(db.Integer)

class Visualization(db.Model):
    __tablename__ = 'visualizations'
    
    id = db.Column(db.Integer, primary_key=True)
    team_id = db.Column(db.Integer, db.ForeignKey('teams.id'), nullable=False)
    game_id = db.Column(db.Integer, db.ForeignKey('games.id'), nullable=True)
    created_by_consultant = db.Column(db.Boolean, default=False)
    is_highlighted = db.Column(db.Boolean, default=False)
    chart_type = db.Column(db.String(50), nullable=False)
    configuration = db.Column(db.JSON)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    # Relationships
    team = db.relationship('Team', backref='visualizations')

# Schemas
class TeamSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Team
        load_instance = True
        exclude = ('password_hash',)

class ConsultantSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Consultant
        load_instance = True
        exclude = ('password_hash',)

class GameSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Game
        load_instance = True
        include_fk = True

class PlayDataSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = PlayData
        load_instance = True
        include_fk = True

class VisualizationSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Visualization
        load_instance = True
        include_fk = True

# Schema instances
team_schema = TeamSchema()
teams_schema = TeamSchema(many=True)
consultant_schema = ConsultantSchema()
consultants_schema = ConsultantSchema(many=True)
game_schema = GameSchema()
games_schema = GameSchema(many=True)
play_data_schema = PlayDataSchema()
play_data_list_schema = PlayDataSchema(many=True)
visualization_schema = VisualizationSchema()
visualizations_schema = VisualizationSchema(many=True)

# Authentication Routes
@app.route('/api/auth/team/register', methods=['POST'])
def register_team():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not all(key in data for key in ['team_name', 'email', 'password']):
            return jsonify({'message': 'Missing required fields'}), 400
        
        # Check if team already exists
        if Team.query.filter_by(email=data['email']).first():
            return jsonify({'message': 'Team with this email already exists'}), 409
        
        # Create new team
        password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        new_team = Team(
            team_name=data['team_name'],
            email=data['email'],
            password_hash=password_hash
        )
        
        db.session.add(new_team)
        db.session.commit()
        
        # Create access token
        access_token = create_access_token(
            identity=str(new_team.id),
            additional_claims={'user_type': 'team', 'user_id': new_team.id}
        )
        
        return jsonify({
            'message': 'Team registered successfully',
            'access_token': access_token,
            'team': team_schema.dump(new_team)
        }), 201
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/auth/team/login', methods=['POST'])
def login_team():
    try:
        data = request.get_json()
        
        if not all(key in data for key in ['email', 'password']):
            return jsonify({'message': 'Missing email or password'}), 400
        
        team = Team.query.filter_by(email=data['email']).first()
        
        if team and bcrypt.check_password_hash(team.password_hash, data['password']):
            access_token = create_access_token(
                identity=str(team.id),
                additional_claims={'user_type': 'team', 'user_id': team.id}
            )
            
            return jsonify({
                'message': 'Login successful',
                'access_token': access_token,
                'team': team_schema.dump(team)
            }), 200
        else:
            return jsonify({'message': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/auth/consultant/register', methods=['POST'])
def register_consultant():
    try:
        data = request.get_json()
        
        if not all(key in data for key in ['name', 'email', 'password']):
            return jsonify({'message': 'Missing required fields'}), 400
        
        if Consultant.query.filter_by(email=data['email']).first():
            return jsonify({'message': 'Consultant with this email already exists'}), 409
        
        password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        new_consultant = Consultant(
            name=data['name'],
            email=data['email'],
            password_hash=password_hash
        )
        
        db.session.add(new_consultant)
        db.session.commit()
        
        access_token = create_access_token(
            identity=str(new_consultant.id),
            additional_claims={'user_type': 'consultant', 'user_id': new_consultant.id}
        )
        
        return jsonify({
            'message': 'Consultant registered successfully',
            'access_token': access_token,
            'consultant': consultant_schema.dump(new_consultant)
        }), 201
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/auth/consultant/login', methods=['POST'])
def login_consultant():
    try:
        data = request.get_json()
        
        if not all(key in data for key in ['email', 'password']):
            return jsonify({'message': 'Missing email or password'}), 400
        
        consultant = Consultant.query.filter_by(email=data['email']).first()
        
        if consultant and bcrypt.check_password_hash(consultant.password_hash, data['password']):
            access_token = create_access_token(
                identity=str(consultant.id),
                additional_claims={'user_type': 'consultant', 'user_id': consultant.id}
            )
            
            return jsonify({
                'message': 'Login successful',
                'access_token': access_token,
                'consultant': consultant_schema.dump(consultant)
            }), 200
        else:
            return jsonify({'message': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/auth/verify', methods=['GET'])
@jwt_required()
def verify_token():
    current_user = get_current_user()
    return jsonify({'user': current_user}), 200

# Game Management Routes
@app.route('/api/games', methods=['POST'])
@jwt_required()
def upload_game():
    try:
        current_user = get_current_user()
        
        # Ensure only teams can upload games
        if current_user['type'] != 'team':
            return jsonify({'message': 'Only teams can upload games'}), 403
        
        # Check if file is in request
        if 'csv_file' not in request.files:
            return jsonify({'message': 'No CSV file provided'}), 400
        
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            return jsonify({'message': 'No file selected'}), 400
        
        # Get form data
        week = request.form.get('week')
        opponent = request.form.get('opponent')
        location = request.form.get('location')
        analytics_focus_notes = request.form.get('analytics_focus_notes', '')
        
        # Validate required fields
        if not all([week, opponent, location]):
            return jsonify({'message': 'Missing required fields: week, opponent, location'}), 400
        
        # Validate location
        if location not in ['Home', 'Away']:
            return jsonify({'message': 'Location must be "Home" or "Away"'}), 400
        
        try:
            week = int(week)
            if week < 1 or week > 17:
                return jsonify({'message': 'Week must be between 1 and 17'}), 400
        except ValueError:
            return jsonify({'message': 'Week must be a number'}), 400
        
        # Read and validate CSV
        csv_content = csv_file.read().decode('utf-8')
        csv_file.seek(0)  # Reset file pointer
        
        try:
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            rows = list(csv_reader)
        except Exception as e:
            return jsonify({'message': f'Invalid CSV format: {str(e)}'}), 400
        
        if not rows:
            return jsonify({'message': 'CSV file is empty'}), 400
        
        # Validate required columns
        required_columns = ['Play ID', 'Down', 'Distance', 'Yard Line', 'Formation', 'Play Type', 'Play Name', 'Result of Play', 'Unit']
        actual_columns = list(rows[0].keys())
        missing_columns = [col for col in required_columns if col not in actual_columns]
        if missing_columns:
            return jsonify({'message': f'Missing required columns: {", ".join(missing_columns)}'}), 400
        
        # Validate data types for numeric columns
        numeric_columns = ['Play ID', 'Yard Line']
        for i, row in enumerate(rows):
            # Check unit type and validate accordingly
            unit = str(row.get('Unit', '')).lower()
            
            # For special teams, skip down/distance validation
            cols_to_validate = numeric_columns.copy()
            if unit not in ['st', 'special teams', 'special']:
                cols_to_validate.extend(['Down', 'Distance'])
            
            for col in cols_to_validate:
                # Allow N/A for special teams down/distance
                if unit in ['st', 'special teams', 'special'] and col in ['Down', 'Distance'] and str(row[col]).upper() == 'N/A':
                    continue
                try:
                    int(row[col])
                except (ValueError, TypeError):
                    return jsonify({'message': f'Row {i+1}: Column "{col}" must be a number, got "{row[col]}"'}), 400
        
        # Create game record
        new_game = Game(
            team_id=current_user['id'],
            week=week,
            opponent=opponent,
            location=location,
            analytics_focus_notes=analytics_focus_notes,
            csv_file_path=f"uploads/team_{current_user['id']}_week_{week}_{opponent}.csv"
        )
        
        db.session.add(new_game)
        db.session.flush()  # Get the game ID
        
        # Save play data
        for row in rows:
            # Extract yards gained from result if available
            yards_gained = 0
            points_scored = 0
            
            # Try to extract yards from result of play
            result_text = str(row['Result of Play']).lower()
            if 'yard' in result_text or 'yd' in result_text:
                yards_match = re.search(r'(\d+)\s*(?:yard|yd)', result_text)
                if yards_match:
                    yards_gained = int(yards_match.group(1))
            
            if 'touchdown' in result_text or 'td' in result_text:
                points_scored = 6
            elif 'field goal' in result_text or 'fg' in result_text:
                points_scored = 3
            
            # Handle down/distance for special teams
            unit = str(row['Unit']).upper()
            down_val = None if str(row['Down']).upper() == 'N/A' else int(row['Down'])
            distance_val = None if str(row['Distance']).upper() == 'N/A' else int(row['Distance'])
            
            play_data = PlayData(
                game_id=new_game.id,
                play_id=int(row['Play ID']),
                down=down_val,
                distance=distance_val,
                yard_line=int(row['Yard Line']),
                formation=str(row['Formation']),
                play_type=str(row['Play Type']),
                play_name=str(row['Play Name']),
                result_of_play=str(row['Result of Play']),
                yards_gained=yards_gained,
                points_scored=points_scored,
                unit=unit
            )
            db.session.add(play_data)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Game uploaded successfully',
            'game': game_schema.dump(new_game),
            'plays_count': len(rows)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Upload failed: {str(e)}'}), 500

@app.route('/api/games', methods=['GET'])
@jwt_required()
def get_games():
    try:
        current_user = get_current_user()
        
        if current_user['type'] == 'team':
            games = Game.query.filter_by(team_id=current_user['id']).order_by(Game.week.desc()).all()
        else:  # consultant
            games = Game.query.order_by(Game.submission_timestamp.desc()).all()
        
        return jsonify({
            'games': games_schema.dump(games)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/games/<int:game_id>', methods=['GET'])
@jwt_required()
def get_game(game_id):
    try:
        current_user = get_current_user()
        
        game = Game.query.get(game_id)
        if not game:
            return jsonify({'message': 'Game not found'}), 404
        
        # Teams can only view their own games
        if current_user['type'] == 'team' and game.team_id != current_user['id']:
            return jsonify({'message': 'Access denied'}), 403
        
        return jsonify({
            'game': game_schema.dump(game)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/games/<int:game_id>/plays', methods=['GET'])
@jwt_required()
def get_game_plays(game_id):
    try:
        current_user = get_current_user()
        
        game = Game.query.get(game_id)
        if not game:
            return jsonify({'message': 'Game not found'}), 404
        
        # Teams can only view their own games
        if current_user['type'] == 'team' and game.team_id != current_user['id']:
            return jsonify({'message': 'Access denied'}), 403
        
        plays = PlayData.query.filter_by(game_id=game_id).order_by(PlayData.play_id).all()
        
        return jsonify({
            'plays': play_data_list_schema.dump(plays),
            'total_plays': len(plays)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Consultant-specific routes
@app.route('/api/consultant/teams', methods=['GET'])
@jwt_required()
def get_teams():
    try:
        current_user = get_current_user()
        
        # Only consultants can access this
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Only consultants can access this endpoint'}), 403
        
        teams = Team.query.all()
        return jsonify({
            'teams': teams_schema.dump(teams)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/consultant/teams/<int:team_id>/games', methods=['GET'])
@jwt_required()
def get_team_games(team_id):
    try:
        current_user = get_current_user()
        
        # Only consultants can access this
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Only consultants can access this endpoint'}), 403
        
        team = Team.query.get(team_id)
        if not team:
            return jsonify({'message': 'Team not found'}), 404
        
        games = Game.query.filter_by(team_id=team_id).order_by(Game.week.desc()).all()
        
        return jsonify({
            'team': team_schema.dump(team),
            'games': games_schema.dump(games)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/consultant/analytics/<int:game_id>', methods=['GET'])
@jwt_required()
def get_game_analytics(game_id):
    try:
        current_user = get_current_user()
        
        # Only consultants can access this
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Only consultants can access this endpoint'}), 403
        
        game = Game.query.get(game_id)
        if not game:
            return jsonify({'message': 'Game not found'}), 404
        
        plays = PlayData.query.filter_by(game_id=game_id).all()
        
        # Calculate basic analytics
        total_plays = len(plays)
        total_yards = sum(play.yards_gained for play in plays)
        total_points = sum(play.points_scored for play in plays)
        
        # Group by play type
        play_type_stats = {}
        for play in plays:
            play_type = play.play_type
            if play_type not in play_type_stats:
                play_type_stats[play_type] = {'count': 0, 'yards': 0, 'avg_yards': 0}
            play_type_stats[play_type]['count'] += 1
            play_type_stats[play_type]['yards'] += play.yards_gained
        
        # Calculate averages
        for play_type in play_type_stats:
            stats = play_type_stats[play_type]
            stats['avg_yards'] = round(stats['yards'] / stats['count'], 2) if stats['count'] > 0 else 0
        
        # Group by formation
        formation_stats = {}
        for play in plays:
            formation = play.formation
            if formation not in formation_stats:
                formation_stats[formation] = {'count': 0, 'yards': 0, 'avg_yards': 0}
            formation_stats[formation]['count'] += 1
            formation_stats[formation]['yards'] += play.yards_gained
        
        # Calculate averages
        for formation in formation_stats:
            stats = formation_stats[formation]
            stats['avg_yards'] = round(stats['yards'] / stats['count'], 2) if stats['count'] > 0 else 0
        
        # Down and distance analysis
        down_stats = {}
        for play in plays:
            down = f"Down {play.down}"
            if down not in down_stats:
                down_stats[down] = {'count': 0, 'yards': 0, 'avg_yards': 0}
            down_stats[down]['count'] += 1
            down_stats[down]['yards'] += play.yards_gained
        
        for down in down_stats:
            stats = down_stats[down]
            stats['avg_yards'] = round(stats['yards'] / stats['count'], 2) if stats['count'] > 0 else 0
        
        return jsonify({
            'game': game_schema.dump(game),
            'summary': {
                'total_plays': total_plays,
                'total_yards': total_yards,
                'total_points': total_points,
                'avg_yards_per_play': round(total_yards / total_plays, 2) if total_plays > 0 else 0
            },
            'play_type_stats': play_type_stats,
            'formation_stats': formation_stats,
            'down_stats': down_stats,
            'plays': play_data_list_schema.dump(plays)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Consultant Data Explorer Routes
@app.route('/api/consultant/team/<int:team_id>/play-data', methods=['GET'])
@jwt_required()
def get_team_play_data(team_id):
    """Get all play data for a team with game context"""
    try:
        current_user = get_current_user()
        
        # Verify consultant access
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Access denied'}), 403
        
        # Get all plays for the team with game information
        plays_query = db.session.query(
            PlayData.id,
            PlayData.play_id,
            PlayData.down,
            PlayData.distance,
            PlayData.yard_line,
            PlayData.formation,
            PlayData.play_type,
            PlayData.play_name,
            PlayData.result_of_play,
            PlayData.yards_gained,
            PlayData.points_scored,
            PlayData.unit,
            PlayData.quarter,
            PlayData.time_remaining,
            PlayData.game_id,
            Game.week.label('game_week'),
            Game.opponent.label('game_opponent')
        ).join(Game).filter(Game.team_id == team_id).all()
        
        # Convert to list of dictionaries
        plays_data = []
        for play in plays_query:
            plays_data.append({
                'id': play.id,
                'play_id': play.play_id,
                'down': play.down,
                'distance': play.distance,
                'yard_line': play.yard_line,
                'formation': play.formation,
                'play_type': play.play_type,
                'play_name': play.play_name,
                'result_of_play': play.result_of_play,
                'yards_gained': play.yards_gained,
                'points_scored': play.points_scored,
                'unit': play.unit,
                'quarter': play.quarter,
                'time_remaining': play.time_remaining,
                'game_id': play.game_id,
                'game_week': play.game_week,
                'game_opponent': play.game_opponent
            })
        
        return jsonify({
            'plays': plays_data,
            'total_plays': len(plays_data)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/consultant/data/filter', methods=['POST'])
@jwt_required()
def filter_play_data():
    """Apply filters to play data and return results"""
    try:
        current_user = get_current_user()
        
        # Verify consultant access
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Access denied'}), 403
        
        data = request.get_json()
        team_id = data.get('team_id')
        filters = data.get('filters', [])
        
        if not team_id:
            return jsonify({'message': 'Team ID is required'}), 400
        
        # Start with base query
        query = db.session.query(
            PlayData.id,
            PlayData.play_id,
            PlayData.down,
            PlayData.distance,
            PlayData.yard_line,
            PlayData.formation,
            PlayData.play_type,
            PlayData.play_name,
            PlayData.result_of_play,
            PlayData.yards_gained,
            PlayData.points_scored,
            PlayData.unit,
            PlayData.quarter,
            PlayData.time_remaining,
            PlayData.game_id,
            Game.week.label('game_week'),
            Game.opponent.label('game_opponent')
        ).join(Game).filter(Game.team_id == team_id)
        
        # Apply filters
        for filter_condition in filters:
            field = filter_condition.get('field')
            operator = filter_condition.get('operator')
            value = filter_condition.get('value')
            
            if not all([field, operator]):
                continue
                
            # Map field names to database columns
            field_mapping = {
                'play_id': PlayData.play_id,
                'down': PlayData.down,
                'distance': PlayData.distance,
                'yard_line': PlayData.yard_line,
                'formation': PlayData.formation,
                'play_type': PlayData.play_type,
                'play_name': PlayData.play_name,
                'result_of_play': PlayData.result_of_play,
                'yards_gained': PlayData.yards_gained,
                'points_scored': PlayData.points_scored,
                'unit': PlayData.unit,
                'quarter': PlayData.quarter,
                'game_week': Game.week,
                'game_opponent': Game.opponent
            }
            
            if field not in field_mapping:
                continue
                
            db_field = field_mapping[field]
            
            # Apply filter based on operator
            if operator == 'equals':
                query = query.filter(db_field == value)
            elif operator == 'not_equals':
                query = query.filter(db_field != value)
            elif operator == 'greater_than':
                query = query.filter(db_field > value)
            elif operator == 'less_than':
                query = query.filter(db_field < value)
            elif operator == 'greater_equal':
                query = query.filter(db_field >= value)
            elif operator == 'less_equal':
                query = query.filter(db_field <= value)
            elif operator == 'contains':
                query = query.filter(db_field.ilike(f'%{value}%'))
            elif operator == 'in' and isinstance(value, list):
                query = query.filter(db_field.in_(value))
        
        # Execute query
        plays_result = query.all()
        
        # Convert to list of dictionaries
        plays_data = []
        for play in plays_result:
            plays_data.append({
                'id': play.id,
                'play_id': play.play_id,
                'down': play.down,
                'distance': play.distance,
                'yard_line': play.yard_line,
                'formation': play.formation,
                'play_type': play.play_type,
                'play_name': play.play_name,
                'result_of_play': play.result_of_play,
                'yards_gained': play.yards_gained,
                'points_scored': play.points_scored,
                'unit': play.unit,
                'quarter': play.quarter,
                'time_remaining': play.time_remaining,
                'game_id': play.game_id,
                'game_week': play.game_week,
                'game_opponent': play.game_opponent
            })
        
        return jsonify({
            'plays': plays_data,
            'total_plays': len(plays_data),
            'filters_applied': len(filters)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/consultant/charts/statistical', methods=['POST'])
@jwt_required()
def generate_statistical_chart():
    """Generate statistical analysis charts"""
    try:
        current_user = get_current_user()
        
        # Verify consultant access
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Access denied'}), 403
        
        data = request.get_json()
        team_id = data.get('team_id')
        chart_type = data.get('chart_type')
        filters = data.get('filters', [])
        chart_options = data.get('options', {})
        
        if not all([team_id, chart_type]):
            return jsonify({'message': 'Team ID and chart type are required'}), 400
        
        # Get filtered play data
        query = db.session.query(
            PlayData.id,
            PlayData.play_id,
            PlayData.down,
            PlayData.distance,
            PlayData.yard_line,
            PlayData.formation,
            PlayData.play_type,
            PlayData.play_name,
            PlayData.result_of_play,
            PlayData.yards_gained,
            PlayData.points_scored,
            PlayData.unit,
            PlayData.quarter,
            PlayData.time_remaining,
            PlayData.game_id,
            Game.week.label('game_week'),
            Game.opponent.label('game_opponent')
        ).join(Game).filter(Game.team_id == team_id)
        
        # Apply filters
        for filter_condition in filters:
            field = filter_condition.get('field')
            operator = filter_condition.get('operator')
            value = filter_condition.get('value')
            
            if not all([field, operator]):
                continue
                
            # Map field names to database columns
            field_mapping = {
                'play_id': PlayData.play_id,
                'down': PlayData.down,
                'distance': PlayData.distance,
                'yard_line': PlayData.yard_line,
                'formation': PlayData.formation,
                'play_type': PlayData.play_type,
                'play_name': PlayData.play_name,
                'result_of_play': PlayData.result_of_play,
                'yards_gained': PlayData.yards_gained,
                'points_scored': PlayData.points_scored,
                'unit': PlayData.unit,
                'quarter': PlayData.quarter,
                'game_week': Game.week,
                'game_opponent': Game.opponent
            }
            
            if field not in field_mapping:
                continue
                
            db_field = field_mapping[field]
            
            # Apply filter based on operator
            if operator == 'equals':
                query = query.filter(db_field == value)
            elif operator == 'not_equals':
                query = query.filter(db_field != value)
            elif operator == 'greater_than':
                query = query.filter(db_field > value)
            elif operator == 'less_than':
                query = query.filter(db_field < value)
            elif operator == 'greater_equal':
                query = query.filter(db_field >= value)
            elif operator == 'less_equal':
                query = query.filter(db_field <= value)
            elif operator == 'contains':
                query = query.filter(db_field.ilike(f'%{value}%'))
            elif operator == 'in' and isinstance(value, list):
                query = query.filter(db_field.in_(value))
        
        # Execute query
        plays_result = query.all()
        
        # Convert to list of dictionaries
        plays_data = []
        for play in plays_result:
            plays_data.append({
                'id': play.id,
                'play_id': play.play_id,
                'down': play.down,
                'distance': play.distance,
                'yard_line': play.yard_line,
                'formation': play.formation,
                'play_type': play.play_type,
                'play_name': play.play_name,
                'result_of_play': play.result_of_play,
                'yards_gained': play.yards_gained,
                'points_scored': play.points_scored,
                'unit': play.unit,
                'quarter': play.quarter,
                'time_remaining': play.time_remaining,
                'game_id': play.game_id,
                'game_week': play.game_week,
                'game_opponent': play.game_opponent
            })
        
        if not plays_data:
            return jsonify({'message': 'No data matches the applied filters'}), 400
        
        # Generate statistical chart
        from footballviz.charts.statistical import create_statistical_chart
        
        try:
            chart_base64 = create_statistical_chart(
                chart_type=chart_type,
                plays_data=plays_data,
                **chart_options
            )
            
            return jsonify({
                'chart_image': chart_base64,
                'chart_type': chart_type,
                'plays_analyzed': len(plays_data),
                'filters_applied': len(filters)
            }), 200
            
        except Exception as chart_error:
            return jsonify({'message': f'Chart generation failed: {str(chart_error)}'}), 500
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/consultant/charts/recommend', methods=['POST']) 
@jwt_required()
def recommend_charts():
    """Recommend chart types based on data characteristics"""
    try:
        current_user = get_current_user()
        
        # Verify consultant access
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Access denied'}), 403
        
        data = request.get_json()
        team_id = data.get('team_id')
        filters = data.get('filters', [])
        selected_plays = data.get('selected_plays', 0)
        
        if not team_id:
            return jsonify({'message': 'Team ID is required'}), 400
        
        # Analyze data characteristics for recommendations
        query = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id)
        
        # Apply filters if any
        for filter_condition in filters:
            field = filter_condition.get('field')
            operator = filter_condition.get('operator')
            value = filter_condition.get('value')
            
            if not all([field, operator]):
                continue
                
            field_mapping = {
                'play_id': PlayData.play_id,
                'down': PlayData.down,
                'distance': PlayData.distance,
                'yard_line': PlayData.yard_line,
                'formation': PlayData.formation,
                'play_type': PlayData.play_type,
                'yards_gained': PlayData.yards_gained,
                'points_scored': PlayData.points_scored,
                'unit': PlayData.unit,
                'quarter': PlayData.quarter
            }
            
            if field in field_mapping:
                db_field = field_mapping[field]
                if operator == 'equals':
                    query = query.filter(db_field == value)
                elif operator == 'greater_than':
                    query = query.filter(db_field > value)
                elif operator == 'less_than':
                    query = query.filter(db_field < value)
                # Add other operators as needed
        
        plays = query.all()
        
        if not plays:
            return jsonify({'recommendations': []}), 200
        
        # Generate recommendations based on data characteristics
        recommendations = []
        
        # Always recommend distribution analysis
        recommendations.append({
            'chart_type': 'distribution',
            'title': 'Yards Distribution Analysis',
            'description': 'Analyze the distribution of yards gained with statistical insights',
            'icon': '📊',
            'priority': 1,
            'reason': 'Shows data distribution patterns and outliers'
        })
        
        # Check for multiple formations
        formations = set(play.formation for play in plays if play.formation)
        if len(formations) > 1:
            recommendations.append({
                'chart_type': 'formation_comparison',
                'title': 'Formation Performance Analysis',
                'description': f'Compare effectiveness across {len(formations)} different formations',
                'icon': '⚡',
                'priority': 2,
                'reason': f'Multiple formations detected ({len(formations)})'
            })
        
        # Check for situational diversity
        downs = set(play.down for play in plays if play.down)
        if len(downs) > 1:
            recommendations.append({
                'chart_type': 'situational',
                'title': 'Situational Analysis',
                'description': 'Analyze performance in different game situations',
                'icon': '🎯',
                'priority': 3,
                'reason': 'Multiple down situations detected'
            })
        
        # Check for field position data
        has_yard_line = any(play.yard_line for play in plays)
        if has_yard_line:
            recommendations.append({
                'chart_type': 'field_heatmap',
                'title': 'Field Position Heatmap',
                'description': 'Visualize play distribution across the field',
                'icon': '🏈',
                'priority': 4,
                'reason': 'Field position data available'
            })
        
        # Check for temporal data
        has_play_sequence = any(play.play_id for play in plays)
        if has_play_sequence and len(plays) > 10:
            recommendations.append({
                'chart_type': 'trends',
                'title': 'Performance Trends',
                'description': 'Track performance changes throughout the game',
                'icon': '📈',
                'priority': 5,
                'reason': 'Sequential play data available'
            })
        
        # Add correlation analysis for sufficient data
        if len(plays) > 20:
            recommendations.append({
                'chart_type': 'correlation',
                'title': 'Variable Correlation Matrix',
                'description': 'Discover relationships between different metrics',
                'icon': '🔗',
                'priority': 6,
                'reason': 'Sufficient data for correlation analysis'
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return jsonify({
            'recommendations': recommendations[:6],  # Limit to top 6
            'data_summary': {
                'total_plays': len(plays),
                'formations': len(formations),
                'avg_yards': sum(play.yards_gained for play in plays) / len(plays) if plays else 0,
                'has_field_position': has_yard_line,
                'has_sequence': has_play_sequence
            }
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Visualization and Highlighting Routes
@app.route('/api/visualizations', methods=['POST'])
@jwt_required()
def create_visualization():
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['team_id', 'chart_type', 'title', 'configuration']
        if not all(field in data for field in required_fields):
            return jsonify({'message': 'Missing required fields'}), 400
        
        # Only consultants can create visualizations for now
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Only consultants can create visualizations'}), 403
        
        # Validate team exists
        team = Team.query.get(data['team_id'])
        if not team:
            return jsonify({'message': 'Team not found'}), 404
        
        visualization = Visualization(
            team_id=data['team_id'],
            game_id=data.get('game_id'),
            created_by_consultant=True,
            is_highlighted=data.get('is_highlighted', False),
            chart_type=data['chart_type'],
            configuration=data['configuration'],
            title=data['title'],
            description=data.get('description', '')
        )
        
        db.session.add(visualization)
        db.session.commit()
        
        return jsonify({
            'message': 'Visualization created successfully',
            'visualization': visualization_schema.dump(visualization)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/visualizations/<int:visualization_id>/highlight', methods=['PUT'])
@jwt_required()
def toggle_highlight(visualization_id):
    try:
        current_user = get_current_user()
        
        # Only consultants can highlight visualizations
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Only consultants can highlight visualizations'}), 403
        
        visualization = Visualization.query.get(visualization_id)
        if not visualization:
            return jsonify({'message': 'Visualization not found'}), 404
        
        visualization.is_highlighted = not visualization.is_highlighted
        db.session.commit()
        
        return jsonify({
            'message': f'Visualization {"highlighted" if visualization.is_highlighted else "unhighlighted"}',
            'visualization': visualization_schema.dump(visualization)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

@app.route('/api/teams/<int:team_id>/visualizations', methods=['GET'])
@jwt_required()
def get_team_visualizations(team_id):
    try:
        current_user = get_current_user()
        
        # Teams can only see their own visualizations, consultants can see all
        if current_user['type'] == 'team' and current_user['id'] != team_id:
            return jsonify({'message': 'Access denied'}), 403
        
        # Get highlighted visualizations for teams, all for consultants
        if current_user['type'] == 'team':
            visualizations = Visualization.query.filter_by(
                team_id=team_id, 
                is_highlighted=True
            ).order_by(Visualization.created_at.desc()).all()
        else:
            visualizations = Visualization.query.filter_by(
                team_id=team_id
            ).order_by(Visualization.created_at.desc()).all()
        
        return jsonify({
            'visualizations': visualizations_schema.dump(visualizations)
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/consultant/visualizations/create-chart', methods=['POST'])
@jwt_required()
def create_chart_from_data():
    try:
        current_user = get_current_user()
        
        # Only consultants can create charts
        if current_user['type'] != 'consultant':
            return jsonify({'message': 'Only consultants can create charts'}), 403
        
        data = request.get_json()
        game_id = data.get('game_id')
        chart_type = data.get('chart_type')
        data_type = data.get('data_type')  # 'play_type', 'formation', 'down'
        
        if not all([game_id, chart_type, data_type]):
            return jsonify({'message': 'Missing required fields: game_id, chart_type, data_type'}), 400
        
        game = Game.query.get(game_id)
        if not game:
            return jsonify({'message': 'Game not found'}), 404
        
        plays = PlayData.query.filter_by(game_id=game_id).all()
        
        # Generate chart data based on data_type
        chart_data = {}
        if data_type == 'play_type':
            for play in plays:
                play_type = play.play_type
                if play_type not in chart_data:
                    chart_data[play_type] = {'count': 0, 'yards': 0}
                chart_data[play_type]['count'] += 1
                chart_data[play_type]['yards'] += play.yards_gained
        elif data_type == 'formation':
            for play in plays:
                formation = play.formation
                if formation not in chart_data:
                    chart_data[formation] = {'count': 0, 'yards': 0}
                chart_data[formation]['count'] += 1
                chart_data[formation]['yards'] += play.yards_gained
        elif data_type == 'down':
            for play in plays:
                down = f"Down {play.down}"
                if down not in chart_data:
                    chart_data[down] = {'count': 0, 'yards': 0}
                chart_data[down]['count'] += 1
                chart_data[down]['yards'] += play.yards_gained
        
        configuration = {
            'data_type': data_type,
            'chart_data': chart_data,
            'game_id': game_id,
            'chart_type': chart_type
        }
        
        title = f"{data_type.replace('_', ' ').title()} Analysis - Week {game.week} vs {game.opponent}"
        
        visualization = Visualization(
            team_id=game.team_id,
            game_id=game_id,
            created_by_consultant=True,
            is_highlighted=data.get('highlight', False),
            chart_type=chart_type,
            configuration=configuration,
            title=title,
            description=f"Analysis of {data_type.replace('_', ' ')} performance"
        )
        
        db.session.add(visualization)
        db.session.commit()
        
        return jsonify({
            'message': 'Chart created successfully',
            'visualization': visualization_schema.dump(visualization),
            'chart_data': chart_data
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 500

# AI Assistant Routes
@app.route('/api/ai/query', methods=['POST'])
@jwt_required()
def ai_query():
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        query = data.get('query', '').lower().strip()
        if not query:
            return jsonify({'message': 'Query is required'}), 400
        
        # Only teams can use AI assistant for now
        if current_user['type'] != 'team':
            return jsonify({'message': 'Only teams can use the AI assistant'}), 403
        
        # Get team's games
        games = Game.query.filter_by(team_id=current_user['id']).all()
        if not games:
            return jsonify({
                'response': "I don't have any game data to analyze yet. Please upload some games first!",
                'query': query
            }), 200
        
        # Parse and respond to predefined queries
        response = process_ai_query(query, games, current_user['id'])
        
        return jsonify({
            'response': response,
            'query': query
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

def process_ai_query(query: str, games: list, team_id: int) -> str:
    """Process advanced AI queries with NLP capabilities"""
    
    import re
    
    # Advanced query processing with synonyms and variations
    query_lower = query.lower()
    
    # Extract opponent and week from query
    opponent = None
    week = None
    
    # Look for opponent name (more flexible matching)
    for game in games:
        opponent_variations = [
            game.opponent.lower(),
            game.opponent.lower().replace(' ', ''),
            game.opponent.split()[0].lower() if ' ' in game.opponent else game.opponent.lower()
        ]
        if any(var in query_lower for var in opponent_variations):
            opponent = game.opponent
            break
    
    # Look for week number (multiple patterns)
    week_patterns = [
        r'week\s+(\d+)',
        r'wk\s+(\d+)',
        r'game\s+(\d+)',
        r'week(\d+)',
        r'w(\d+)'
    ]
    for pattern in week_patterns:
        week_match = re.search(pattern, query_lower)
        if week_match:
            week = int(week_match.group(1))
            break
    
    # Find the specific game
    target_game = None
    if opponent and week:
        target_game = next((g for g in games if g.opponent.lower() == opponent.lower() and g.week == week), None)
    elif opponent:
        target_game = next((g for g in games if g.opponent.lower() == opponent.lower()), None)
    elif week:
        target_game = next((g for g in games if g.week == week), None)
    
    # Enhanced pattern matching for queries
    
    # Yards queries (multiple variations)
    yards_patterns = ['total yards', 'yards gained', 'yards', 'yardage', 'offensive yards']
    if any(pattern in query_lower for pattern in yards_patterns):
        if target_game:
            plays = PlayData.query.filter_by(game_id=target_game.id).all()
            total_yards = sum(play.yards_gained for play in plays)
            return f"In the game against {target_game.opponent} in week {target_game.week}, your team gained a total of {total_yards} yards."
        else:
            # All games total
            all_plays = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id).all()
            total_yards = sum(play.yards_gained for play in all_plays)
            return f"Across all your games, your team has gained a total of {total_yards} yards."
    
    # Plays queries (multiple variations)
    elif any(pattern in query_lower for pattern in ['total plays', 'how many plays', 'number of plays', 'play count', 'plays run']):
        if target_game:
            plays = PlayData.query.filter_by(game_id=target_game.id).all()
            return f"In the game against {target_game.opponent} in week {target_game.week}, your team ran {len(plays)} total plays."
        else:
            all_plays = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id).all()
            return f"Across all your games, your team has run {len(all_plays)} total plays."
    
    # Points scored queries
    elif 'points' in query and ('scored' in query or 'score' in query):
        if target_game:
            plays = PlayData.query.filter_by(game_id=target_game.id).all()
            total_points = sum(play.points_scored for play in plays)
            return f"In the game against {target_game.opponent} in week {target_game.week}, your team scored {total_points} points."
        else:
            all_plays = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id).all()
            total_points = sum(play.points_scored for play in all_plays)
            return f"Across all your games, your team has scored {total_points} total points."
    
    # Average yards per play queries
    elif 'average yards' in query and 'play' in query:
        play_type = None
        if 'run' in query:
            play_type = 'Run'
        elif 'pass' in query:
            play_type = 'Pass'
        
        if target_game:
            plays = PlayData.query.filter_by(game_id=target_game.id)
            if play_type:
                plays = plays.filter_by(play_type=play_type)
            plays = plays.all()
            
            if plays:
                avg_yards = sum(play.yards_gained for play in plays) / len(plays)
                play_type_text = f" for {play_type} plays" if play_type else ""
                return f"In the game against {target_game.opponent} in week {target_game.week}, your team averaged {avg_yards:.2f} yards per play{play_type_text}."
            else:
                return f"No {play_type.lower() if play_type else ''} plays found for that game."
        else:
            plays_query = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id)
            if play_type:
                plays_query = plays_query.filter(PlayData.play_type == play_type)
            plays = plays_query.all()
            
            if plays:
                avg_yards = sum(play.yards_gained for play in plays) / len(plays)
                play_type_text = f" for {play_type} plays" if play_type else ""
                return f"Across all your games, your team has averaged {avg_yards:.2f} yards per play{play_type_text}."
            else:
                return f"No {play_type.lower() if play_type else ''} plays found."
    
    # Best formation query
    elif 'best formation' in query or 'most effective formation' in query:
        plays_query = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id)
        if target_game:
            plays_query = plays_query.filter(Game.id == target_game.id)
        plays = plays_query.all()
        
        formation_stats = {}
        for play in plays:
            if play.formation not in formation_stats:
                formation_stats[play.formation] = {'yards': 0, 'count': 0}
            formation_stats[play.formation]['yards'] += play.yards_gained
            formation_stats[play.formation]['count'] += 1
        
        if formation_stats:
            best_formation = max(formation_stats.items(), 
                               key=lambda x: x[1]['yards'] / x[1]['count'] if x[1]['count'] > 0 else 0)
            avg_yards = best_formation[1]['yards'] / best_formation[1]['count']
            context = f" in the game against {target_game.opponent}" if target_game else " across all games"
            return f"Your most effective formation{context} is {best_formation[0]}, averaging {avg_yards:.2f} yards per play."
        else:
            return "No formation data available."
    
    # Run vs Pass efficiency
    elif ('run vs pass' in query or 'pass vs run' in query or 'run or pass' in query) and ('efficient' in query or 'effective' in query or 'more' in query):
        plays_query = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id)
        if target_game:
            plays_query = plays_query.filter(Game.id == target_game.id)
        plays = plays_query.all()
        
        run_plays = [p for p in plays if p.play_type == 'Run']
        pass_plays = [p for p in plays if p.play_type == 'Pass']
        
        run_avg = sum(p.yards_gained for p in run_plays) / len(run_plays) if run_plays else 0
        pass_avg = sum(p.yards_gained for p in pass_plays) / len(pass_plays) if pass_plays else 0
        
        context = f" in the game against {target_game.opponent}" if target_game else " across all games"
        
        if run_avg > pass_avg:
            return f"Your running game is more efficient{context}. Run plays average {run_avg:.2f} yards vs {pass_avg:.2f} yards for pass plays."
        elif pass_avg > run_avg:
            return f"Your passing game is more efficient{context}. Pass plays average {pass_avg:.2f} yards vs {run_avg:.2f} yards for run plays."
        else:
            return f"Your run and pass games are equally efficient{context}, both averaging around {run_avg:.2f} yards per play."
    
    # Advanced analytics queries
    elif any(pattern in query_lower for pattern in ['trends', 'improvement', 'getting better', 'worse', 'progress']):
        if len(games) >= 2:
            # Compare first and last game performance
            first_game = min(games, key=lambda g: g.week)
            last_game = max(games, key=lambda g: g.week)
            
            first_plays = PlayData.query.filter_by(game_id=first_game.id).all()
            last_plays = PlayData.query.filter_by(game_id=last_game.id).all()
            
            first_avg = sum(p.yards_gained for p in first_plays) / len(first_plays) if first_plays else 0
            last_avg = sum(p.yards_gained for p in last_plays) / len(last_plays) if last_plays else 0
            
            if last_avg > first_avg:
                improvement = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
                return f"Your team is improving! Your average yards per play increased from {first_avg:.2f} in week {first_game.week} to {last_avg:.2f} in week {last_game.week} - that's a {improvement:.1f}% improvement!"
            elif first_avg > last_avg:
                decline = ((first_avg - last_avg) / first_avg * 100) if first_avg > 0 else 0
                return f"Your team's performance has declined from {first_avg:.2f} yards per play in week {first_game.week} to {last_avg:.2f} in week {last_game.week} - that's a {decline:.1f}% decrease. Let's analyze what changed."
            else:
                return f"Your team's performance has been consistent, averaging around {first_avg:.2f} yards per play."
        else:
            return "I need at least 2 games to analyze trends. Upload more game data!"
    
    # Analytics focus interpretation
    elif any(pattern in query_lower for pattern in ['focus', 'coach wants', 'notes', 'priority', 'emphasis']):
        focus_insights = []
        for game in games:
            if game.analytics_focus_notes:
                focus_insights.append(f"Week {game.week} vs {game.opponent}: {game.analytics_focus_notes}")
        
        if focus_insights:
            return f"Here are your coach's analytics priorities:\\n\\n" + "\\n".join(f"• {insight}" for insight in focus_insights)
        else:
            return "No specific analytics focus notes found in your uploaded games."
    
    # Situational analysis
    elif any(pattern in query_lower for pattern in ['red zone', 'redzone', 'goal line', 'short yardage']):
        # Analyze plays near the goal line (within 20 yards)
        all_plays = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id).all()
        red_zone_plays = [p for p in all_plays if p.yard_line >= 80]  # Assuming 100-yard field
        
        if red_zone_plays:
            total_yards = sum(p.yards_gained for p in red_zone_plays)
            touchdowns = len([p for p in red_zone_plays if p.points_scored >= 6])
            success_rate = (touchdowns / len(red_zone_plays)) * 100 if red_zone_plays else 0
            
            return f"Red zone analysis: {len(red_zone_plays)} plays, {touchdowns} touchdowns ({success_rate:.1f}% success rate), {total_yards} total yards. Average: {total_yards/len(red_zone_plays):.2f} yards per play."
        else:
            return "No red zone plays found in your data."
    
    # Third down analysis
    elif 'third down' in query_lower or '3rd down' in query_lower:
        all_plays = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id).all()
        third_down_plays = [p for p in all_plays if p.down == 3]
        
        if third_down_plays:
            successful = len([p for p in third_down_plays if p.yards_gained >= p.distance])
            success_rate = (successful / len(third_down_plays)) * 100
            avg_yards = sum(p.yards_gained for p in third_down_plays) / len(third_down_plays)
            
            return f"Third down performance: {successful}/{len(third_down_plays)} conversions ({success_rate:.1f}% success rate), averaging {avg_yards:.2f} yards per attempt."
        else:
            return "No third down plays found in your data."
    
    # Comparison queries
    elif any(pattern in query_lower for pattern in ['compare', 'versus', 'vs', 'difference between']):
        if len(games) >= 2:
            games_sorted = sorted(games, key=lambda g: g.week)
            game1, game2 = games_sorted[0], games_sorted[-1]
            
            plays1 = PlayData.query.filter_by(game_id=game1.id).all()
            plays2 = PlayData.query.filter_by(game_id=game2.id).all()
            
            yards1 = sum(p.yards_gained for p in plays1)
            yards2 = sum(p.yards_gained for p in plays2)
            
            return f"Comparison: Week {game1.week} vs {game1.opponent}: {yards1} yards. Week {game2.week} vs {game2.opponent}: {yards2} yards. Difference: {yards2 - yards1} yards."
        else:
            return "I need at least 2 games to make comparisons."
    
    # Weakness analysis
    elif any(pattern in query_lower for pattern in ['weakness', 'weaknesses', 'problem', 'struggle', 'worst']):
        all_plays = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id).all()
        
        # Analyze by formation
        formation_stats = {}
        for play in all_plays:
            if play.formation not in formation_stats:
                formation_stats[play.formation] = {'yards': 0, 'count': 0}
            formation_stats[play.formation]['yards'] += play.yards_gained
            formation_stats[play.formation]['count'] += 1
        
        if formation_stats:
            worst_formation = min(formation_stats.items(), 
                                key=lambda x: x[1]['yards'] / x[1]['count'] if x[1]['count'] > 0 else float('inf'))
            avg_yards = worst_formation[1]['yards'] / worst_formation[1]['count']
            
            return f"Your biggest weakness appears to be the {worst_formation[0]} formation, averaging only {avg_yards:.2f} yards per play. Consider adjusting this formation or using it less frequently."
        else:
            return "Not enough data to identify weaknesses."
    
    # Default response - try local AI if available, fallback to predefined responses
    else:
        # Try local AI first
        if local_ai.is_available():
            all_plays = db.session.query(PlayData).join(Game).filter(Game.team_id == team_id).all()
            plays_data = [
                {
                    'yards_gained': play.yards_gained,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'down': play.down,
                    'distance': play.distance,
                    'points_scored': play.points_scored
                }
                for play in all_plays
            ]
            
            ai_response = local_ai.analyze_football_data(query, plays_data)
            return ai_response
        
        # Fallback to predefined responses
        advanced_queries = [
            "What were the total yards against [Opponent]?",
            "How many plays did we run in week [X]?",
            "What is our best/worst formation?",
            "Is our run or pass game more efficient?",
            "How are we trending/improving?",
            "What are our coach's focus areas?",
            "How is our red zone performance?",
            "What's our third down conversion rate?",
            "What are our weaknesses?",
            "Compare our games"
        ]
        
        return f"I understand natural language! Here are some things you can ask me:\\n\\n" + "\\n".join(f"• {q}" for q in advanced_queries) + "\\n\\n💡 Tip: Install Ollama for enhanced AI responses!"

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# AI status check
@app.route('/api/ai/status', methods=['GET'])
def ai_status():
    ollama_available = local_ai.is_available()
    available_models = local_ai.get_available_models() if ollama_available else []
    
    return jsonify({
        'ollama_available': ollama_available,
        'available_models': available_models,
        'current_model': local_ai.model if ollama_available else None,
        'recommendation': 'Install Ollama for enhanced AI responses' if not ollama_available else 'Local AI ready!'
    }), 200

# Real-time collaboration API endpoints
@app.route('/api/collaboration/sessions', methods=['GET'])
@jwt_required()
def get_active_sessions():
    return jsonify(collaboration_service.get_active_sessions()), 200

@app.route('/api/collaboration/notify', methods=['POST'])
@jwt_required()
def send_notification():
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        target_user_id = data.get('target_user_id')
        notification_type = data.get('type', 'general')
        message = data.get('message', '')
        
        if not target_user_id or not message:
            return jsonify({'message': 'Missing required fields'}), 400
        
        collaboration_service.send_notification(target_user_id, {
            'type': notification_type,
            'message': message,
            'from_user': {
                'id': current_user['id'],
                'type': current_user['type']
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return jsonify({'message': 'Notification sent successfully'}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Advanced Reporting API endpoints
@app.route('/api/reports/team/<int:team_id>', methods=['GET'])
@jwt_required()
def generate_team_report(team_id):
    try:
        current_user = get_current_user()
        
        # Check permissions
        if current_user['type'] == 'team' and current_user['id'] != team_id:
            return jsonify({'message': 'Access denied'}), 403
        
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        format_type = request.args.get('format', 'pdf')  # pdf or excel
        
        if format_type not in ['pdf', 'excel']:
            return jsonify({'message': 'Invalid format. Use pdf or excel'}), 400
        
        # Generate report
        report_buffer = report_generator.generate_team_performance_report(
            team_id, start_date, end_date, format_type
        )
        
        # Prepare response
        team = Team.query.get(team_id)
        filename = f"{team.team_name}_performance_report_{datetime.now().strftime('%Y%m%d')}"
        
        if format_type == 'pdf':
            mimetype = 'application/pdf'
            filename += '.pdf'
        else:
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename += '.xlsx'
        
        return send_file(
            io.BytesIO(report_buffer.getvalue()),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/reports/consultant/<int:consultant_id>', methods=['POST'])
@jwt_required()
def generate_consultant_report(consultant_id):
    try:
        current_user = get_current_user()
        
        # Only consultants can generate their own reports
        if current_user['type'] != 'consultant' or current_user['id'] != consultant_id:
            return jsonify({'message': 'Access denied'}), 403
        
        data = request.get_json()
        team_ids = data.get('team_ids', [])
        format_type = data.get('format', 'pdf')
        
        if not team_ids:
            return jsonify({'message': 'Team IDs required'}), 400
        
        if format_type not in ['pdf', 'excel']:
            return jsonify({'message': 'Invalid format. Use pdf or excel'}), 400
        
        # Generate report
        report_buffer = report_generator.generate_consultant_report(
            consultant_id, team_ids, format_type
        )
        
        # Prepare response
        consultant = Consultant.query.get(consultant_id)
        filename = f"{consultant.name}_consultant_report_{datetime.now().strftime('%Y%m%d')}"
        
        if format_type == 'pdf':
            mimetype = 'application/pdf'
            filename += '.pdf'
        else:
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename += '.xlsx'
        
        return send_file(
            io.BytesIO(report_buffer.getvalue()),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/exports/game-data/<int:game_id>', methods=['GET'])
@jwt_required()
def export_game_data(game_id):
    try:
        current_user = get_current_user()
        
        # Get game and verify permissions
        game = Game.query.get(game_id)
        if not game:
            return jsonify({'message': 'Game not found'}), 404
        
        if current_user['type'] == 'team' and current_user['id'] != game.team_id:
            return jsonify({'message': 'Access denied'}), 403
        
        format_type = request.args.get('format', 'csv')  # csv, json, excel
        
        if format_type == 'csv':
            return export_game_csv(game)
        elif format_type == 'json':
            return export_game_json(game)
        elif format_type == 'excel':
            return export_game_excel(game)
        else:
            return jsonify({'message': 'Invalid format. Use csv, json, or excel'}), 400
            
    except Exception as e:
        return jsonify({'message': str(e)}), 500

def export_game_csv(game):
    """Export game data as CSV"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        'Play ID', 'Down', 'Distance', 'Yard Line', 'Formation', 'Play Type',
        'Play Name', 'Result of Play', 'Yards Gained', 'Points Scored'
    ])
    
    # Data
    for play in game.play_data:
        writer.writerow([
            play.play_id, play.down, play.distance, play.yard_line,
            play.formation, play.play_type, play.play_name,
            play.result_of_play, play.yards_gained, play.points_scored
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"game_{game.week}_{game.opponent.replace(' ', '_')}_data.csv"
    )

def export_game_json(game):
    """Export game data as JSON"""
    game_data = {
        'game_info': {
            'week': game.week,
            'opponent': game.opponent,
            'location': game.location,
            'analytics_focus_notes': game.analytics_focus_notes,
            'submission_timestamp': game.submission_timestamp.isoformat() if game.submission_timestamp else None
        },
        'plays': []
    }
    
    for play in game.play_data:
        game_data['plays'].append({
            'play_id': play.play_id,
            'down': play.down,
            'distance': play.distance,
            'yard_line': play.yard_line,
            'formation': play.formation,
            'play_type': play.play_type,
            'play_name': play.play_name,
            'result_of_play': play.result_of_play,
            'yards_gained': play.yards_gained,
            'points_scored': play.points_scored
        })
    
    return send_file(
        io.BytesIO(json.dumps(game_data, indent=2).encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        download_name=f"game_{game.week}_{game.opponent.replace(' ', '_')}_data.json"
    )

def export_game_excel(game):
    """Export game data as Excel"""
    buffer = io.BytesIO()
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = f"Week {game.week} vs {game.opponent}"
    
    # Game info
    sheet['A1'] = f"Week {game.week} vs {game.opponent} ({game.location})"
    sheet['A1'].font = openpyxl.styles.Font(size=16, bold=True)
    sheet.merge_cells('A1:J1')
    
    # Headers
    headers = [
        'Play ID', 'Down', 'Distance', 'Yard Line', 'Formation', 'Play Type',
        'Play Name', 'Result of Play', 'Yards Gained', 'Points Scored'
    ]
    
    for col, header in enumerate(headers, 1):
        cell = sheet.cell(row=3, column=col, value=header)
        cell.font = openpyxl.styles.Font(bold=True)
        cell.fill = openpyxl.styles.PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
    
    # Data
    for row, play in enumerate(game.play_data, 4):
        data = [
            play.play_id, play.down, play.distance, play.yard_line,
            play.formation, play.play_type, play.play_name,
            play.result_of_play, play.yards_gained, play.points_scored
        ]
        
        for col, value in enumerate(data, 1):
            sheet.cell(row=row, column=col, value=value)
    
    # Auto-adjust column widths
    for col in range(1, len(headers) + 1):
        sheet.column_dimensions[chr(64 + col)].width = 15
    
    workbook.save(buffer)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f"game_{game.week}_{game.opponent.replace(' ', '_')}_data.xlsx"
    )

# ================================
# LANGCHAIN ENHANCED AI ENDPOINTS
# ================================

@app.route('/api/langchain/status', methods=['GET'])
@jwt_required()
def langchain_status():
    """Get LangChain service status and capabilities"""
    try:
        stats = langchain_service.get_service_stats()
        workflows = analysis_pipeline.get_available_workflows()
        examples = query_translator.get_query_examples()
        
        return jsonify({
            'status': 'available' if stats['is_available'] else 'unavailable',
            'service_stats': stats,
            'available_workflows': workflows,
            'query_examples': examples,
            'capabilities': {
                'natural_language_queries': True,
                'multi_step_analysis': True,
                'conversation_memory': True,
                'query_translation': True
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/langchain/query', methods=['POST'])
@jwt_required()
def natural_language_query():
    """Process natural language football queries"""
    try:
        current_user_info = get_current_user()
        if not current_user_info:
            return jsonify({'error': 'Authentication failed'}), 401
        
        data = request.get_json()
        query = data.get('query')
        game_id = data.get('game_id')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get plays data
        if game_id:
            # Specific game
            plays_query = PlayData.query.filter_by(game_id=game_id)
            if current_user_info['type'] == 'team':
                game = Game.query.filter_by(id=game_id, team_id=current_user_info['user_id']).first()
                if not game:
                    return jsonify({'error': 'Game not found or access denied'}), 403
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        else:
            # All user's games
            if current_user_info['type'] == 'team':
                plays_query = PlayData.query.join(Game).filter(Game.team_id == current_user_info['user_id'])
            else:
                plays_query = PlayData.query.all()
            
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        
        if not plays_data:
            return jsonify({'error': 'No game data available'}), 404
        
        # Process with LangChain
        result = langchain_service.conversational_query(query, plays_data)
        
        return jsonify({
            'success': True,
            'result': result,
            'data_count': len(plays_data),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Natural language query error: {str(e)}")
        return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

@app.route('/api/langchain/translate', methods=['POST'])
@jwt_required()
def translate_query():
    """Translate natural language to SQL filters"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Translate query
        translation_result = query_translator.translate_query(query)
        
        # Analyze query difficulty
        difficulty_analysis = query_translator.analyze_query_difficulty(query)
        
        return jsonify({
            'success': translation_result.success,
            'filters': translation_result.filters,
            'confidence_score': translation_result.confidence_score,
            'error_message': translation_result.error_message,
            'suggested_corrections': translation_result.suggested_corrections,
            'difficulty_analysis': difficulty_analysis,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/langchain/analyze', methods=['POST'])
@jwt_required()
def enhanced_analysis():
    """Enhanced football data analysis with LangChain"""
    try:
        current_user_info = get_current_user()
        if not current_user_info:
            return jsonify({'error': 'Authentication failed'}), 401
        
        data = request.get_json()
        query = data.get('query')
        game_id = data.get('game_id')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get plays data (similar to natural_language_query)
        if game_id:
            plays_query = PlayData.query.filter_by(game_id=game_id)
            if current_user_info['type'] == 'team':
                game = Game.query.filter_by(id=game_id, team_id=current_user_info['user_id']).first()
                if not game:
                    return jsonify({'error': 'Game not found or access denied'}), 403
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        else:
            if current_user_info['type'] == 'team':
                plays_query = PlayData.query.join(Game).filter(Game.team_id == current_user_info['user_id'])
            else:
                plays_query = PlayData.query.all()
            
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        
        if not plays_data:
            return jsonify({'error': 'No game data available'}), 404
        
        # Enhanced analysis with LangChain
        analysis = langchain_service.analyze_football_data_enhanced(query, plays_data)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'data_count': len(plays_data),
            'conversation_length': len(langchain_service.get_conversation_history()),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Enhanced analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/langchain/workflow', methods=['POST'])
@jwt_required()
def execute_workflow():
    """Execute predefined analysis workflow"""
    try:
        current_user_info = get_current_user()
        if not current_user_info:
            return jsonify({'error': 'Authentication failed'}), 401
        
        data = request.get_json()
        workflow_name = data.get('workflow_name')
        game_id = data.get('game_id')
        
        if not workflow_name:
            return jsonify({'error': 'Workflow name is required'}), 400
        
        # Get plays data
        if game_id:
            plays_query = PlayData.query.filter_by(game_id=game_id)
            if current_user_info['type'] == 'team':
                game = Game.query.filter_by(id=game_id, team_id=current_user_info['user_id']).first()
                if not game:
                    return jsonify({'error': 'Game not found or access denied'}), 403
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        else:
            if current_user_info['type'] == 'team':
                plays_query = PlayData.query.join(Game).filter(Game.team_id == current_user_info['user_id'])
            else:
                plays_query = PlayData.query.all()
            
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        
        if not plays_data:
            return jsonify({'error': 'No game data available'}), 404
        
        # Execute workflow
        pipeline_result = analysis_pipeline.execute_workflow(workflow_name, plays_data)
        
        return jsonify({
            'success': pipeline_result.success,
            'pipeline_id': pipeline_result.pipeline_id,
            'steps': [
                {
                    'step_id': step.step_id,
                    'step_type': step.step_type.value,
                    'success': step.success,
                    'insights': step.insights,
                    'metrics': step.metrics,
                    'error_message': step.error_message,
                    'execution_time': step.execution_time,
                    'timestamp': step.timestamp
                }
                for step in pipeline_result.steps
            ],
            'summary': pipeline_result.summary,
            'recommendations': pipeline_result.recommendations,
            'total_execution_time': pipeline_result.total_execution_time,
            'timestamp': pipeline_result.timestamp
        }), 200
        
    except Exception as e:
        logging.error(f"Workflow execution error: {str(e)}")
        return jsonify({'error': f'Workflow execution failed: {str(e)}'}), 500

@app.route('/api/langchain/multi-step', methods=['POST'])
@jwt_required()
def multi_step_analysis():
    """Execute multi-step analysis with custom queries"""
    try:
        current_user_info = get_current_user()
        if not current_user_info:
            return jsonify({'error': 'Authentication failed'}), 401
        
        data = request.get_json()
        queries = data.get('queries', [])
        game_id = data.get('game_id')
        
        if not queries or not isinstance(queries, list):
            return jsonify({'error': 'List of queries is required'}), 400
        
        # Get plays data
        if game_id:
            plays_query = PlayData.query.filter_by(game_id=game_id)
            if current_user_info['type'] == 'team':
                game = Game.query.filter_by(id=game_id, team_id=current_user_info['user_id']).first()
                if not game:
                    return jsonify({'error': 'Game not found or access denied'}), 403
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        else:
            if current_user_info['type'] == 'team':
                plays_query = PlayData.query.join(Game).filter(Game.team_id == current_user_info['user_id'])
            else:
                plays_query = PlayData.query.all()
            
            plays_data = [
                {
                    'play_id': play.play_id,
                    'down': play.down,
                    'distance': play.distance,
                    'yard_line': play.yard_line,
                    'formation': play.formation,
                    'play_type': play.play_type,
                    'play_name': play.play_name,
                    'result_of_play': play.result_of_play,
                    'yards_gained': play.yards_gained,
                    'points_scored': play.points_scored
                }
                for play in plays_query.all()
            ]
        
        if not plays_data:
            return jsonify({'error': 'No game data available'}), 404
        
        # Execute multi-step analysis
        results = langchain_service.multi_step_analysis(queries, plays_data)
        
        return jsonify({
            'success': True,
            'results': results,
            'data_count': len(plays_data),
            'total_steps': len(queries),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Multi-step analysis error: {str(e)}")
        return jsonify({'error': f'Multi-step analysis failed: {str(e)}'}), 500

@app.route('/api/langchain/conversation/history', methods=['GET'])
@jwt_required()
def get_conversation_history():
    """Get conversation history"""
    try:
        history = langchain_service.get_conversation_history()
        return jsonify({
            'success': True,
            'history': history,
            'length': len(history),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/langchain/conversation/clear', methods=['POST'])
@jwt_required()
def clear_conversation_history():
    """Clear conversation history"""
    try:
        langchain_service.clear_conversation_history()
        return jsonify({
            'success': True,
            'message': 'Conversation history cleared',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/langchain/workflows', methods=['GET'])
@jwt_required()
def get_workflows():
    """Get available analysis workflows"""
    try:
        workflows = analysis_pipeline.get_available_workflows()
        return jsonify({
            'success': True,
            'workflows': workflows,
            'count': len(workflows),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)