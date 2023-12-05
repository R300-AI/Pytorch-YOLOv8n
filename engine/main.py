import torch, argparse, os, subprocess, yaml
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')

    #指向./dataset，並進行模型訓練
    with open('./dataset/data.yaml', 'r') as f:
        data =yaml.safe_load(f)
        data['train'], data['test'], data['val'] = '/usr/src/ultralytics/dataset/train/images', '/usr/src/ultralytics/dataset/test/images', '/usr/src/ultralytics/dataset/valid/images'
    with open('./dataset/data.yaml', 'w') as f:
        yaml.dump(data, f)
    results = model.train(data='./dataset/data.yaml', epochs=2)

    #輸出模型並存放於./output
    saved_path = model.export(format='tflite', half=True, int8=True)
    output_name = saved_path.split('/')[-1]
    subprocess.run(["cp",  saved_path.replace(output_name, "best_float32.tflite"), "./output/FLOAT32.tflite"])
    subprocess.run(["cp",  saved_path.replace(output_name, "best_float16.tflite"), "./output/FLOAT16.tflite"])
    subprocess.run(["cp",  saved_path.replace(output_name, "best_int8.tflite"), "./output/INT8.tflite"])
    subprocess.run(["cp",  saved_path.replace(output_name, "best_full_integer_quant.tflite"), "./output/FLOAT32_quant.tflite"])  #CPU / GPU
    subprocess.run(["cp",  saved_path.replace(output_name, "best_integer_quant.tflite"), "./output/INT8_quant.tflite"])  #CPU / TPU / NPU
    print('Performing finished')