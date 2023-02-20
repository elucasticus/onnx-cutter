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

CMD python3 onnx_manager_flask.py --operation run_all --onnx_file cifar10.onnx --onnx_path cifar10 --image_batch images --image_size_x 32 --image_size_y 32 --image_is_grayscale False --server_url http://172.17.0.2:5000/onnx

