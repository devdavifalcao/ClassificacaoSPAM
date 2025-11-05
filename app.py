from flask import Flask, request, jsonify, send_from_directory
import re

app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.route("/")
def index():

    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    
    app.run(debug=True)