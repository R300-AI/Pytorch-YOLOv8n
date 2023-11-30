import yaml, os, shutil

def Resize(PROJECT_PATH, input_size):
    images_list = os.listdir(PROJECT_PATH + '/train/images')
    for image_name in images_list:
        target = PROJECT_PATH + '/train/images/' + image_name
        cv2.imwrite(target, cv2.resize(cv2.imread(target), (input_size, input_size), interpolation=cv2.INTER_AREA))
    print('Resized', len(images_list), 'Images(Training Subset)')