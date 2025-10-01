#!/usr/bin/env python3

from app import app

if __name__ == '__main__':
    with app.app_context():
        from app import db
        db.create_all()
        print("Database initialized successfully")
    
    print("Starting Flask server on http://0.0.0.0:5001")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)