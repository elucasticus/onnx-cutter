import numpy as np
import requests
import time

def main():
    x = np.random.randn(3, 1).flatten()
    data = {'x': x.tolist()}
    departure_time = time.time()
    response = requests.post("http://127.0.0.1:5000/test", json=data).json()
    arrival_time = response["arrival_time"]
    uploading_time = arrival_time - departure_time
    print(uploading_time)

if __name__ == "__main__":
    main()


