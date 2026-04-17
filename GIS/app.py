from flask import Flask, send_from_directory
import os
import subprocess

app = Flask(__name__)

# Run the dashboard script once to ensure index.html exists when starting
def generate_dashboard():
    try:
        print("Generating latest dashboard data...")
        subprocess.run(["python", "aqi_dashboard.py"], check=True)
    except Exception as e:
        print(f"Error generating dashboard: {e}")

@app.route("/")
def index():
    # If index.html doesn't exist, try to generate it
    if not os.path.exists("index.html"):
        generate_dashboard()
    return send_from_directory(".", "index.html")

@app.route("/update")
def update():
    """Endpoint to manually trigger a data update on the server"""
    generate_dashboard()
    return "Dashboard updated successfully!"

if __name__ == "__main__":
    # Ensure it's generated at least once
    if not os.path.exists("index.html"):
        generate_dashboard()
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
