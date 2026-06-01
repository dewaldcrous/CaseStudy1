"""
Build executable for the Traffic Light Optimisation Demo
=========================================================

Prerequisites:
    pip install pyinstaller

Run:
    python build_exe.py

This creates:
    dist/TrafficDemo.exe  (standalone executable)
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("  Building Traffic Light Optimisation Executable")
    print("=" * 60)
    print()

    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"  PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("  PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # Single executable
        "--windowed",                   # No console window
        "--name", "TrafficDemo",        # Output name
        "--add-data", "src;src",        # Include src folder
        "--add-data", "models;models",  # Include models folder
        "--hidden-import", "sklearn.ensemble._forest",
        "--hidden-import", "sklearn.tree._utils",
        "--hidden-import", "lightgbm",
        "--hidden-import", "matplotlib.backends.backend_tkagg",
        "run_demo.py"
    ]

    print("  Running PyInstaller...")
    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.check_call(cmd)
        print()
        print("=" * 60)
        print("  BUILD SUCCESSFUL!")
        print("  Executable: dist/TrafficDemo.exe")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"  Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
