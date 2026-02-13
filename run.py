# Startup Script

import subprocess
import sys
import webbrowser
import time
import threading

def install_dependencies():
    print("📦 Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
    ])
    print("✅ Dependencies installed")

def open_browser():
    time.sleep(2)
    webbrowser.open("http://localhost:8080")

def main():
    print("=" * 40)
    print("  SmartData Optimizer - Starting...")
    print("=" * 40)
    print()
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"⚠️ Warning: Could not install dependencies: {e}")
    
    print()
    print("🚀 Starting server at http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Import and run the FastAPI app (now uses lightweight app.py)
    import uvicorn
    uvicorn.run("smartOp.app:app", host="0.0.0.0", port=8080, reload=False)

if __name__ == "__main__":
    main()
