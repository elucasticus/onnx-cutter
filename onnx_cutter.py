from onnx_splitter import onnx_model_split_all

def main():
    onnx_file = "cifar10.onnx"
    onnx_model_split_all(onnx_file)

if __name__ == "__main__" :
    main()
    
