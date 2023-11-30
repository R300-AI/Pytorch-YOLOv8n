from ultralytics import YOLO
from .utils import make_dirs
import subprocess, os

class YOLOv8Builder():
    def __init__(self, args):
        self.model = YOLO('yolov8n.pt')
        self.train(args)
        
    def train(self, args):
        self.results = self.model.train(data='datasets/' + args.PROJECT_NAME + "/data.yaml", epochs=args.EPOCHS, patience=args.EARLY_STOP, verbose=True)

    def export(self):
        #train history
        output_path = "/usr/src/ultralytics/outputs/" + os.environ['DATASET_FORMAT'] + '/' + os.environ['PROJECT_NAME']
        make_dirs(output_path)
        subprocess.run(["cp", '-r',  self.results.save_dir, output_path + '/train_history'])

        #.tflite model
        saved_model_path = self.model.export(format='tflite')
        return saved_model_path, output_path
