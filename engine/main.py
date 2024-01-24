import torch, argparse, os, subprocess, yaml
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')
    print(subprocess.run(["ls"]))
    #指向./dataset，並進行模型訓練
    with open('./dataset/data.yaml', 'r') as f:
        data =yaml.safe_load(f)
        data['train'], data['test'], data['val'] = '/usr/src/ultralytics/dataset/train/images', '/usr/src/ultralytics/dataset/test/images', '/usr/src/ultralytics/dataset/valid/images'
    with open('./dataset/data.yaml', 'w') as f:
        yaml.dump(data, f)
    results = model.train(data='./dataset/data.yaml', epochs=2)

    #輸出模型並存放於./benchmark
    saved_path = model.export(format='tflite', half=True, int8=True)
    output_name = saved_path.split('/')[-1]
    subprocess.run(["cp",  saved_path.replace(output_name, "best_float32.tflite"), "./benchmark/FLOAT32.tflite"])
    subprocess.run(["cp",  saved_path.replace(output_name, "best_float16.tflite"), "./benchmark/FLOAT16.tflite"])
    subprocess.run(["cp",  saved_path.replace(output_name, "best_int8.tflite"), "./benchmark/INT8.tflite"])
    subprocess.run(["cp",  saved_path.replace(output_name, "best_full_integer_quant.tflite"), "./benchmark/FLOAT32_quant.tflite"])  #CPU / GPU
    subprocess.run(["cp",  saved_path.replace(output_name, "best_integer_quant.tflite"), "./benchmark/INT8_quant.tflite"])  #CPU / TPU / NPU
    print('Performing finished')
