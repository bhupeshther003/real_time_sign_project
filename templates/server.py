# simple_server.py
from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 5000
Handler = SimpleHTTPRequestHandler

with HTTPServer(("", PORT), Handler) as server:
    print(f"Serving HTTP on port {PORT}")
    server.serve_forever()
