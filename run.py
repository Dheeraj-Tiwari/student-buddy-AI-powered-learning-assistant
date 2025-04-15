# run.py
import os
# Add the project root to the Python path to help with imports if needed
import sys
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.app import app # This should work now with the __init__.py or standard package structure

if __name__ == "__main__":
    # Use 0.0.0.0 to make it accessible from other devices on your network
    # Use 127.0.0.1 (default) for local access only
    host = "127.0.0.1"
    port = 5000
    debug = False # Keep debug=True for development, set to False for production
    print(f"Starting Flask server at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)