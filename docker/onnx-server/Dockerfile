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

CMD python3 -m flask --app 'onnx_server:run("onnx_models/test", "CPU", None)' run --host=0.0.0.0
