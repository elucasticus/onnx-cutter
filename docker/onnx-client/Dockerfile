FROM python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
# RUN apt-get install python3-setuptools
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
# RUN pip3 install --extra-index-url https://alpine-wheels.github.io/index numpy
RUN pip3 install numpy
RUN pip3 install -r requirements.txt

COPY . .

CMD python3 onnx_manager_flask.py --operation run_all --onnx_file onnx_models/test/mobilenet.onnx --onnx_path onnx_models/test --image_batch images/mobilenet --image_size_x 160 --image_size_y 160 --image_is_grayscale False --server_url http://192.168.86.21:5000/onnx

