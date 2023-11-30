class YOLOv8_augmentatation():
    def __init__(self):
        self.support_matrics = {'rotate': 0}
        '''self.support_matrics = {'rotate':0, 'shear':0, 'grayscale':0, 'hue':0, 'saturation':0, 'brightness':0, 'exposure':0, 'noise':0, 'cutout':0,
                                'mosaic':0, 'box_flip':0, 'box_ratate':0, 'box_crop':0, 'box_rotation':0, 'box_shear':0, 'box_brightness':0, 
                                'box_exposure':0, 'box_blur':0, 'box_noise':0}'''

    def load(self, metadata):
        for ops in self.support_matrics:
            if ops in metadata.keys():
                self.support_matrics[ops] = metadata[ops]
            else:
                metadata[ops] = 0
        return metadata
        
    def get_ops(self):
        return self.support_matrics

    def fit(self, target_path, recipe):
        print('Recipe:', recipe)
        for ops in self.support_matrics:
            if recipe[ops] == 1 and self.support_matrics[ops] != 1:
                print(ops, 'Augmented(', target_path, ')')
                self.support_matrics[ops] = 1
        return self.support_matrics

class YOLOv8_Processor():
    def CheckFormat(source_path):
        source_dir = os.listdir(source_path)
        format_error = False
        if 'data.yaml' in source_dir and 'test' in source_dir and 'train' in source_dir and 'valid' in source_dir:
            for subset in  ['train', 'test', 'valid']:
                subset_dir = os.listdir(source_path + '/' + subset)
                if 'images' not in subset_dir or 'labels' not in subset_dir:
                    print('Error: subset', subset, "doesn't have 'images/' or '/labels', please check your folder format."); format_error = True; break
            with open(source_path + '/data.yaml', 'r') as f:
                data = yaml.safe_load(f)
                if 'names' not in data.keys():
                    print('Error: attribute names no in data.yaml, please config label_name list of your dataset.'); format_error = True
        else:
            print("Error: Project format incorrect, please check your folder have 'data.yaml', 'train/', 'test/', 'val/', "); format_error = True
        return format_error

    def Images_HealthCare(target_path, input_shape):
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

    def Annotes_HealthCare(target_path, names, input_shape):
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