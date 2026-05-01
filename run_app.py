import subprocess
import sys
import time
import os

def main():
    print("Starting FastAPI Backend...")
    # Start the FastAPI server using uvicorn
    # Use python -m uvicorn so we use the current environment
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # Wait a moment for the API to initialize
    time.sleep(3)
    
    print("Starting Streamlit Frontend...")
    # Start the Streamlit app
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    try:
        # Wait for both processes
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        api_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
