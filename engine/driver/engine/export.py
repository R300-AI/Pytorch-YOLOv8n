from ultralytics import YOLO
import subprocess, os

class YOLOv8Exporter():
    def __init__(self, args, saved_model_path, output_path):
        if args.PRECISION == 16:
            subprocess.run(["cp",  saved_model_path.replace('32', '16'), output_path + "/YOLOv8n_FLOAT16_BATCH"+ str(args.BATCH_SIZE) + ".tflite"])
        elif args.PRECISION == 32:
            subprocess.run(["cp",  saved_model_path, output_path + "/YOLOv8n_FLOAT32_BATCH"+ str(args.BATCH_SIZE) + ".tflite"])
        print(saved_model_path)
        subprocess.run(["ls"])
        subprocess.run(["cp", '-r', 'YOLOv8.py', output_path + '/process.py'])