from edgevision import Vision2D_Config
import torch, argparse, os

if torch.cuda.is_available():
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--Option", default='yolov8n', type=str, help="Options: [yolov8n]")
args = parser.parse_args()

if __name__ == '__main__':
    #load model
    #perform training