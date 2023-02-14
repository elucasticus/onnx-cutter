import numpy as np
import requests
import time
from onnx_manager_flask import onnx_run_complete

def main():
    # x = np.random.randn(3, 1).flatten()
    # data = {'x': x.tolist()}
    # departure_time = time.time()
    # response = requests.post("http://127.0.0.1:5000/test", json=data).json()
    # arrival_time = response["arrival_time"]
    # uploading_time = arrival_time - departure_time
    # print(uploading_time)

    onnx_path = "cifar10"
    split_layer = "onnx::Flatten_16"
    onnx_run_complete(onnx_path, split_layer, None, "images", 32, 32, False, "AMD64", "CPU", "CPU_FP64")

if __name__ == "__main__":
    main()


