from flask import Flask, request
import time

app = Flask(__name__)

@app.route("/status")
def return_server_status():
    return "<h1>Operational</h1>"

@app.route("/test", methods=["POST", "GET"])
def test():
    data = request.json
    arrival_time = time.time()
    x = data["x"]
    print(x)
    return {"Outcome": "Success", "arrival_time": arrival_time}

