import os
import jwt
import requests
from functools import wraps
from flask import request, jsonify, g
from supabase import create_client, Client
from gotrue import SyncGoTrueClient
from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
SUPABASE_JWT_SECRET = os.getenv('SUPABASE_JWT_SECRET')

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_JWT_SECRET]):
    raise ValueError("Missing Supabase configuration. Please set SUPABASE_URL, SUPABASE_SERVICE_KEY, and SUPABASE_JWT_SECRET environment variables.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def verify_supabase_token(token):
    """
    Verify a Supabase JWT token and return user information
    """
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=['HS256'],
            audience='authenticated'
        )

        # Extract user information
        user_id = payload.get('sub')
        email = payload.get('email')
        role = payload.get('user_metadata', {}).get('role')

        if not user_id:
            return None

        return {
            'id': user_id,
            'email': email,
            'role': role,
            'payload': payload
        }
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception as e:
        print(f"Token verification error: {e}")
        return None

def get_user_profile(user_id, role):
    """
    Get user profile from Supabase based on role
    """
    try:
        if role == 'team':
            result = supabase.table('teams').select('*').eq('id', user_id).single().execute()
        elif role == 'consultant':
            result = supabase.table('consultants').select('*').eq('id', user_id).single().execute()
        else:
            return None

        if result.data:
            return result.data
        return None
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return None

def require_auth(f):
    """
    Decorator to require authentication for API endpoints
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]  # Remove 'Bearer ' prefix
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        # Verify the token
        user_data = verify_supabase_token(token)
        if not user_data:
            return jsonify({'error': 'Token is invalid or expired'}), 401

        # Store user data in Flask's g object
        g.current_user = user_data

        # Optionally fetch full profile
        if user_data.get('role'):
            profile = get_user_profile(user_data['id'], user_data['role'])
            g.current_user['profile'] = profile

        return f(*args, **kwargs)

    return decorated

def require_role(required_role):
    """
    Decorator to require specific role for API endpoints
    """
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            user_role = g.current_user.get('role')
            if user_role != required_role:
                return jsonify({'error': f'Access denied. Required role: {required_role}'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

def get_current_user():
    """
    Get current user from Flask's g object
    """
    return g.get('current_user')

# Helper functions for specific roles
def require_team_auth(f):
    """
    Decorator requiring team authentication
    """
    return require_role('team')(f)

def require_consultant_auth(f):
    """
    Decorator requiring consultant authentication
    """
    return require_role('consultant')(f)

def create_user_profile(user_id, email, profile_data, role):
    """
    Create user profile in appropriate table
    """
    try:
        profile_data['id'] = user_id
        profile_data['email'] = email

        if role == 'team':
            result = supabase.table('teams').insert(profile_data).execute()
        elif role == 'consultant':
            result = supabase.table('consultants').insert(profile_data).execute()
        else:
            return None

        return result.data
    except Exception as e:
        print(f"Error creating user profile: {e}")
        return None