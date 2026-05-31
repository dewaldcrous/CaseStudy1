#!/usr/bin/env python3
"""
Simple HTTP server to serve the demo files.
Run this script, then open http://localhost:8000/demo.html in your browser.
"""
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000

# Change to the solution directory
os.chdir(Path(__file__).parent)

Handler = http.server.SimpleHTTPRequestHandler

print(f"Starting server at http://localhost:{PORT}")
print(f"Open http://localhost:{PORT}/demo.html in your browser")
print("Press Ctrl+C to stop the server")

# Try to open browser automatically
try:
    webbrowser.open(f"http://localhost:{PORT}/demo.html")
except:
    pass

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
