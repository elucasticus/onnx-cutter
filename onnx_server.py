from flask import Flask, request
import time
from onnx_second_inference_flask import onnx_search_and_run_second_half
import numpy as np

app = Flask(__name__)

@app.route("/status")
def return_server_status():
    return "<h1>Operational</h1>"

@app.route("/test", methods=["POST", "GET"])
def test():
    data = request.json
    arrival_time = time.time()
    onnx_path = "cifar10"
    returnData = onnx_search_and_run_second_half(onnx_path, None, data, None, "CPU", "CPU_FP64")
    returnData["Outcome"] = "Success"
    returnData["arrival_time"] = arrival_time
    returnData["result"] = returnData["result"].tolist()
    return returnData

