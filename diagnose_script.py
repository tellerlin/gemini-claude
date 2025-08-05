#!/usr/bin/env python3
"""
Diagnostic Script - Checks for main.py import issues (Updated for src structure)
"""
import sys
import os
import traceback
from pathlib import Path

def check_files_exist():
    print("\nChecking required files:")
    # Check for files in the src directory
    src_path = Path('src')
    if not src_path.is_dir():
        print(f"  Error: `src` directory not found!")
        return

    required_files = [
        src_path / 'main.py',
        src_path / 'config.py',
        src_path / 'anthropic_api.py',
        Path('requirements.txt'),
        Path('Dockerfile')
    ]
    for file_path in required_files:
        exists = file_path.exists()
        print(f"  {file_path}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"    Error: Missing file {file_path}")

def check_imports():
    print("\nChecking third-party library imports:")
    # Updated list of libraries, removed litellm
    third_party_libs = [
        'fastapi', 'uvicorn', 'pydantic', 'loguru',
        'google.generativeai', 'dotenv', 'cachetools'
    ]
    for lib in third_party_libs:
        try:
            __import__(lib.split('.')[0])
            print(f"  {lib}: ✓")
        except ImportError as e:
            print(f"  {lib}: ✗ - {e}")

def check_main_module():
    print("\nChecking main application module (src.main):")
    # Add src to python path for local import check
    sys.path.insert(0, os.getcwd())
    try:
        # Try to import the main module from src
        from src import main as app_main
        print("  Importing src.main module: ✓")

        if hasattr(app_main, 'app'):
            print("  Found 'app' attribute: ✓")
        else:
            print("  Found 'app' attribute: ✗")
    except ImportError as e:
        print(f"  Importing src.main module: ✗ - {e}")
        traceback.print_exc()
    finally:
        # Clean up path
        if sys.path[0] == os.getcwd():
            sys.path.pop(0)

# (Other functions like check_python_version, check_environment_variables can remain the same)
# --- main execution part needs to call the updated functions ---
def main():
    print("=== Gemini Claude Adapter Diagnostic Tool (Updated) ===\n")
    check_files_exist()
    check_imports()
    # check_local_imports() # This is effectively covered by check_main_module now
    check_main_module()
    print("\n=== Diagnostics Complete ===")

if __name__ == "__main__":
    main()
