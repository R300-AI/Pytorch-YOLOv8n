import yaml, os, subprocess, cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from edgevision.utils.management import Make_Directory
from .process import YOLOv8_augmentatation

class Detect_Dataset():
    def __init__(self, format):
        print('Config已設定為Detect任務類型')
        if format == 'YOLOv8':
            self.processor = YOLOv8_augmentatation()
        else:
            print('format', format, 'are not support for dectection yet.')

    def Augmentatation(self, target_path):
        #只有在為現有專案擴增版本時才會根據recipe.yaml執行資料增強
        if '/Benchmark' not in target_path:
            with open(target_path + '/data.yaml', 'r') as f:
                metadata = yaml.safe_load(f)
            with open(target_path + '/recipe.yaml', 'r') as f:
                recipe = yaml.safe_load(f)
            metadata['Augs'] = self.processor.fit(target_path, recipe)
            with open(target_path + "/recipe.yaml", 'w') as f:
                yaml.dump(metadata['Augs'], f)

    def Generate_Directory(self, source_path, target_path, args):
        print('[Dataset Configuration]\n - Source:{source}\n - Target:{target}'.format(source=source_path, target=target_path))

        if '/Benchmark' in target_path:
            #建立新的專案，並提供recipe.yaml增強選項。在原本的專案中必須存在data.yaml的names屬性(資料集類別名稱的list)
            Make_Directory(target_path.replace('/Benchmark', ''))
            with open(source_path + '/data.yaml', 'r') as f:
                data = yaml.safe_load(f)
            metadata = {'dtype': args.DATA_TYPE,  'task': args.TASK,  'format': args.FORMAT,  'project_name': args.PROJECT_NAME, 'version': 'Benchmark', "description":"(None)",
                        'names': data['names'], 'input_shape': 640, 'images':0, 'annoted':0, 'annotations':{name:0 for name in data['names']}}
            metadata['Augs'] = self.processor.get_ops()
            subprocess.run(["cp", '-r',  source_path, target_path])
            with open(target_path + "/data.yaml", 'w') as f:
                yaml.dump(metadata, f)
            with open(target_path + "/recipe.yaml", 'w') as f:
                yaml.dump(metadata['Augs'], f)
        else:
            #為現有專案擴增版本，更新可供擴增的選項
            with open(source_path + '/data.yaml', 'r') as f:
                metadata = yaml.safe_load(f)
            metadata['Augs'] = self.processor.load(metadata['Augs'])
            metadata['version'] = target_path.split('/')[-1]
            subprocess.run(["cp", '-r',  source_path, target_path])
            with open(target_path + "/data.yaml", 'w') as f:
                yaml.dump(metadata, f)

    def Summary(self, target_path):
        with open(target_path + '/data.yaml', 'r') as f:
                metadata = yaml.safe_load(f)
        #彙整影像,標籤與data.yaml設定檔
        sample_size = {}
        for subset in  ['train', 'test', 'valid']:
            image_list, annote_list = os.listdir(target_path + '/{subset}/images'.format(subset=subset)), os.listdir(target_path + '/{subset}/labels'.format(subset=subset))
            sample_size[subset] = len(image_list)
            metadata['images'] += sample_size[subset]
            metadata['annoted'] += len(set(['.'.join(i.split('.')[:-1]) for i in image_list]).intersection(set(['.'.join(i.split('.')[:-1]) for i in annote_list])))
        metadata['train_size'], metadata['test_size'], metadata['val_size'] = round(sample_size['train']/ metadata['images'], 2), round(sample_size['test']/ metadata['images'], 2), round(sample_size['valid']/ metadata['images'], 2)
        metadata['train'], metadata['test'], metadata['val'] = target_path+ '/train/images', target_path + '/test/images', target_path + '/valid/images'

        for subset in  ['train', 'test', 'valid']:
            label_list = os.listdir(target_path + '/{subset}/labels'.format(subset=subset))
            for label_path in os.listdir(target_path + '/{subset}/labels'.format(subset=subset)):
                with open(target_path + '/{subset}/labels/'.format(subset=subset) + label_path) as labels:
                    for label in labels.readlines():
                        metadata['annotations'][metadata['names'][int(np.array(label.replace('\n', '').split(' ')).astype(float)[0])]] += 1
        
        self.Images_HealthCare(target_path, metadata['input_shape'])
        self.Annotes_HealthCare(target_path, metadata['names'], metadata['input_shape'])

        with open(target_path + "/data.yaml", 'w') as f:
            yaml.dump(metadata, f)
        print(metadata)
        print('Config file saved to', (target_path + "/data.yaml").split("ultralytics/")[-1])
        print('Bbox distribution array saved to', (target_path + "/bbox_distribution.dat").split("ultralytics/")[-1], 'and .png')

    def CheckDataFormat(self, source_path):
        source_dir = os.listdir(source_path)
        error = True
        if 'data.yaml' in source_dir and 'test' in source_dir and 'train' in source_dir and 'valid' in source_dir:
            for subset in  ['train', 'test', 'valid']:
                subset_dir = os.listdir(source_path + '/' + subset)
                if 'images' not in subset_dir or 'labels' not in subset_dir:
                    print('Error: subset', subset, "doesn't have 'images/' or '/labels', please check your folder format."); error = False; break
            with open(source_path + '/data.yaml', 'r') as f:
                data = yaml.safe_load(f)
                if 'names' not in data.keys():
                    print('Error: attribute names no in data.yaml, please config label_name list of your dataset.'); error = False
        else:
            print("Error: Project format incorrect, please check your folder have 'data.yaml', 'train/', 'test/', 'val/', "); error = False
        return error


    def Images_HealthCare(self, target_path, input_shape):
        shape_distribution = []
        for subset in  ['train', 'test', 'valid']:
            for image_path in os.listdir(target_path + '/{subset}/images'.format(subset=subset)):
                img = cv2.imread(target_path + '/{subset}/images/'.format(subset=subset) + image_path)
                shape_distribution.append(img.shape[:2])
        shape_distribution = np.array(shape_distribution)

        max_ticks = int(max([np.max(shape_distribution), input_shape]) * 1.05)
        fig = plt.figure(figsize=(12, 12)); ax = fig.add_subplot(1, 1, 1)
        ax.plot([0, input_shape], [input_shape, input_shape], color='red', alpha=0.3)
        ax.plot([input_shape, input_shape], [0, input_shape], color='red', alpha=0.3)
        ax.fill_between([0, max_ticks], [0, 0], [0, int(max_ticks * 0.382)], color='gray', alpha=0.3)
        ax.fill_between([0, int(max_ticks * 0.382)], [0, max_ticks], [max_ticks, max_ticks], color='gray', alpha=0.3)
        ax.scatter(shape_distribution[:, 0], shape_distribution[:, 1])
        plt.xlim([0, max_ticks]); plt.ylim([0, max_ticks]); plt.xticks([]); plt.yticks([]); plt.savefig(target_path + "/shape_distribution.png")
        shape_distribution.tofile(target_path + "/shape_distribution.dat")

    def Annotes_HealthCare(self, target_path, names, input_shape):
        bbox_distribution = np.zeros((len(names), input_shape, input_shape)) #class, x_dim, y_dim
        for subset in  ['train', 'test', 'valid']:
            for label_path in os.listdir(target_path + '/{subset}/labels'.format(subset=subset)):
                with open(target_path + '/{subset}/labels'.format(subset=subset) + '/' + label_path) as labels:
                    for label in labels.readlines():
                        label = np.array(label.replace('\n', '').split(' ')).astype(float)
                        name = int( label[0]); label *= input_shape
                        bbox_distribution[name][int(label[1] - label[3]/2):int(label[1] + label[3]/2), int(label[2] - label[4]/2):int(label[2] + label[4]/2)] += 1

        #輸出data.yaml設定檔、分布的原始資料與視覺化圖表(bbox_distribution.dat, bbox_distribution.png)
        for i, name in enumerate(names):
            fig = plt.figure(figsize=(12, 12)); ax = fig.add_subplot(1, 1, 1)
            sns.heatmap(bbox_distribution[i].T, ax=ax, cbar=False, fmt="d")
            plt.xticks([]); plt.yticks([]); plt.savefig(target_path + "/" + name + ".png")
        bbox_distribution.tofile(target_path + "/bbox_distribution.dat")