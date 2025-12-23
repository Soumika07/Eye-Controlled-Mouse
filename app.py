from flask import Flask, render_template, jsonify
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start-system")
def start_system():
    # IMPORTANT: run ONLY one combined file
    subprocess.Popen(["python", "backend_eye_mouse.py"])
    return jsonify({"status": "System Started"})

if __name__ == "__main__":
    app.run(debug=True)
