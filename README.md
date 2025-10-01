# Football Analytics Platform - Backend

Python Flask backend API for the Football Analytics SaaS platform.

## Tech Stack

- **Framework**: Flask 2.3.3
- **Database**: PostgreSQL (Production), SQLite (Development)
- **Authentication**: JWT + Supabase Auth
- **ORM**: SQLAlchemy
- **Real-time**: Socket.IO
- **AI/ML**: LangChain, Ollama, scikit-learn
- **Deployment**: Railway

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL (for production)
- Supabase account

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:
```
SECRET_KEY=your-secret-key-32-chars-minimum
JWT_SECRET_KEY=your-jwt-secret-32-chars-minimum
DATABASE_URL=postgresql://user:pass@host:5432/dbname
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_JWT_SECRET=your-jwt-secret-from-supabase
FRONTEND_URL=https://your-frontend.vercel.app
ENVIRONMENT=production
PORT=5000
```

### Development

```bash
python app.py
```

Runs on http://localhost:5000

### Production

```bash
gunicorn --bind 0.0.0.0:$PORT app:app --workers 4 --timeout 120
```

## Project Structure

```
├── app.py                      # Main Flask application
├── ai_local.py                 # Local AI integration
├── analysis_pipeline.py        # Football analysis pipeline
├── collaboration.py            # Real-time collaboration
├── reporting.py                # Report generation
├── langchain_service.py        # LangChain integration
├── nl_query_translator.py      # Natural language query processing
├── footballviz_api.py          # Visualization API
├── supabase_auth.py           # Supabase authentication
├── jwt_helper.py              # JWT utilities
├── data/                       # Data files
├── decision_service/           # Decision analysis service
├── footballviz/                # Visualization module
├── scripts/                    # Utility scripts
└── tests/                      # Test files
```

## API Endpoints

### Authentication
- `POST /api/auth/team/register` - Team registration
- `POST /api/auth/team/login` - Team login
- `POST /api/auth/consultant/register` - Consultant registration
- `POST /api/auth/consultant/login` - Consultant login
- `GET /api/auth/verify` - Token verification

### Game Data
- `POST /api/games` - Upload game data
- `GET /api/teams/{id}/games` - Get team games
- `GET /api/games/{id}` - Get game details

### Analytics
- `POST /api/analysis/query` - Natural language query
- `GET /api/analysis/insights` - Get AI insights

### Collaboration
- WebSocket endpoints for real-time features

### Health
- `GET /api/health` - Health check endpoint

## Database Setup

### Development (SQLite)
Database automatically created on first run.

### Production (PostgreSQL)

1. Create database:
```sql
CREATE DATABASE football_analytics;
```

2. Run migrations (if using Alembic):
```bash
flask db upgrade
```

Or initialize tables manually on first request.

## Deployment

### Railway

1. Create new project on Railway
2. Connect GitHub repository
3. Add PostgreSQL database service
4. Configure environment variables
5. Deploy

Railway configuration in `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "gunicorn --bind 0.0.0.0:$PORT app:app --workers 4 --timeout 120",
    "healthcheckPath": "/api/health"
  }
}
```

## Security Notes

- Never commit `.env` files
- Use strong random values for SECRET_KEY and JWT_SECRET_KEY
- Generate secrets: `python -c "import secrets; print(secrets.token_hex(32))"`
- Keep SUPABASE_SERVICE_KEY secure (never expose to frontend)
- Configure CORS to only allow your frontend domain

## Related Repositories

- **Frontend**: [football-analytics-frontend](https://github.com/yourusername/football-analytics-frontend)

## License

Private - All rights reserved