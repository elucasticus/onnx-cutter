from numpy.core.numeric import True_
#import sclblonnx as so
import onnx
#from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
#from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import datetime
import numpy as np
import argparse
import time
import json
import os
import csv
from onnx_first_inference import onnx_run_first_half
import pickle
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import onnxruntime
#from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantFormat, QuantType
from onnx_opcounter import calculate_params, calculate_macs
from onnx2json import convert
from onnx_splitter import (onnx_model_split, onnx_model_split_all, onnx_model_multi_split, onnx_model_early_exits_split, 
                           onnx_model_split_all_singlenode, onnx_show_graph)
from onnx_utils import *
from packaging import version
import copy
import onnx_tool
from onnx_tool import create_ndarray_f32

#Dependencies for flask
import requests

def onnx_run_complete(onnx_path, split_layer, image_file, image_batch, img_size_x, img_size_y, is_grayscale, 
                      platform, exec_provider, device_type, xml_file=None):
  '''
  Run a complete cycle of inference, meaning run the first half of the model locally, get the results, load them on the cloud, 
  execute the second part of the model on the cloud and get the relative results.

  :param onnx_path: the path to the collection of models were to find the correct one to use for the inference
  :param split_layer: the name of the DNN layer where the split has been performed on the model choosen
  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :param platform: the platform where the script is executed, in order to use the right client for MinIO, OSCAR and Kubernetes
  :param exec_provider: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device_type: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  :return: the 1° inference execution time, the 2° inference execution time, the 2° inference OSCAR execution time and the 2nd inference Kubernetes pod execution time
  '''
  #Default Argument Values
  if is_grayscale == None: is_grayscale = False
  if platform == None: platform = "AMD64"

  #TESTING
  dictTensors = {} # global dictTensors 

  #Iterate through the subdirectories to find the ONNX model splitted at our selected layer
  onnx_file = None
  if split_layer == "NO_SPLIT" or split_layer == "PROFILING": 
    #Since we skip the local execution and don't use splits, the full model is required instead of the onnx_path
    onnx_file = onnx_path
  else:
    split_layer = split_layer.replace("/", '-').replace(":", '_')
    print("Search for: " + split_layer)
    for dir in os.listdir(onnx_path):
      if dir.find("_on_") > 0:
        index = dir.index('_on_')
        d = dir[index+4:]
        #print("Check: " + d)
        if d == split_layer:
          print("Found Layer: " + d)
          onnx_file = onnx_path + "/" + dir + "/first_half.onnx"
          break
  
  # Only proceed if an onnx file is found for this layer
  if onnx_file != None:
    print("\n ###Start the 1st Inference Execution Locally\n")

    #Load the Onnx Model    --    Got to do it just to have the input tensor shape for data_processing
    model_onnx = load_onnx_model(onnx_file)

    # Process input data (image or batch of images)
    inputData = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale, model_onnx.graph.input[0])

    # Check if we have to SKIP the 1st Inference Execution Locally
    if split_layer == "NO_SPLIT" or split_layer == "PROFILING":
      print("\n ###SKIP the 1st Inference Execution Locally, run directly the whole model on the Cloud..\n")
      print(" Create a results.txt file with the whole image instead of the tensor..")
      data = {
        "splitLayer": "NO_SPLIT",
        "fullModelFile": onnx_file,
        "execTime1": 0,   #1st Inference Execution Time
        "result": inputData,
        "tensorLength": inputData.size,
        "tensorSaveTime": 0
      }

      # Profiling case must be differentiated
      if split_layer == "PROFILING":
        data["splitLayer"] = "PROFILING"

      #Save the first input tensor (input)
      dictTensors[model_onnx.graph.input[0].name] = inputData    #"first" won't be recognized, use the input of the model

      # Save the Tensor on a file
      with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(data, f)
    else:
      #Run at Inference the First part of the ONNX DNN Model (on single image OR batch)
      print("onnx_file: ", onnx_file)
      resData, _ = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=True, xml_file=xml_file)       #Now getting time with profiling (20/08/22)
      #I don't need to save the file since it's already saved in onnx_first_inference.py 

      #Save the Intermediate Tensors
      print("Saving the intermediate tensor...")
      dictTensors[resData["splitLayer"]] = resData["result"]
      data = resData
    
    data["result"] = data["result"].tolist()
    #Send the Intermediate Tensors to the server
    print("Sending the intermediate tensors to the server...")
    departure_time = time.time()
    response = requests.post("http://127.0.0.1:5000/test", json=data).json()

    #Compute uploading time
    arrival_time = response["arrival_time"]
    uploading_time = arrival_time - departure_time
    print("uploading_time: %f" %uploading_time)

    #Print the results
    print("#### 1st Inference Execution Time: " + str(response["execTime1"]) + " s")
    print("#### 2nd Inference Execution Time: " + str(response["execTime2"]) + " s")
    print("---------------------------------------------------")
    print("#### Tensor Length: " + str(response["tensorLength"]))
    print("#### 1st Inf. Tensor Save Time: " + str(response["tensorSaveTime"]) + " s")
    print("#### Networking Time: " + str(uploading_time) + " s")
    print("#### 2nd Inf. Tensor Load Time: " + str(response["tensorLoadTime"]) + " s")

    if split_layer == "PROFILING":
      return response["execTime1"], response["execTime2"], response["tensorLength"], response["tensorSaveTime"], response["tensorLoadTime"], uploading_time, response["profilingTableCloud"]
    else: 
      return response["execTime1"], response["execTime2"], response["tensorLength"], response["tensorSaveTime"], response["tensorLoadTime"], uploading_time
  return -1,-1,-1,-1,-1,-1
    


