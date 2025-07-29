#!/usr/bin/env python3
"""
Gemini Claude Adapter - Entry Point
This script provides a simple way to run the adapter for development/testing.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point for development/testing"""
    print("🚀 Gemini Claude Adapter")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("src/main.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Error: Virtual environment not found")
        print("Please create it with: python3 -m venv venv")
        sys.exit(1)
    
    # Activate virtual environment and run the server
    uvicorn_path = venv_path / "bin" / "uvicorn"
    if not uvicorn_path.exists():
        print("❌ Error: uvicorn not found in virtual environment")
        print("Please install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Set environment variables for development
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', str(Path.cwd()))
    
    print("🔧 Starting development server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Run uvicorn with the correct module path
        subprocess.run([
            str(uvicorn_path),
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()