import os
from shutil import copyfile
import numpy as np

ratio = 0.3
dest = '../train_val_dataset'

image_path = os.path.join('../../data/images')
label_path = os.path.join('../../data/label_info_fields')

train_dir = os.path.join(dest, 'train')
test_dir = os.path.join(dest, 'test')

train_dir_img = os.path.join(train_dir, 'images')
train_dir_label = os.path.join(train_dir, 'labels')

test_dir_img = os.path.join(test_dir, 'images')
test_dir_label = os.path.join(test_dir, 'labels')

if not os.path.exists(dest):
    os.makedirs(dest)

    os.makedirs(train_dir)
    os.makedirs(train_dir_img)
    os.makedirs(train_dir_label)

    os.makedirs(test_dir)
    os.makedirs(test_dir_img)
    os.makedirs(test_dir_label)

def main():
    # image_files = np.array(next(os.walk(image_path), (None, None, []))[2])  # [] if no file
    # label_files = np.array(next(os.walk(label_corner_path), (None, None, []))[2])  # [] if no file
    # label_info_files = np.array(next(os.walk(label_info_path), (None, None, []))[2])  # [] if no file

    image_files = np.array([x for x in os.listdir(image_path) if x.endswith(".jpg")])
    label_files = np.array([x.replace(".jpg",".xml") for x in image_files])
    num_images = image_files.shape[0]

    x = None
    if num_images < 10000:
        x = np.random.permutation(num_images)
    else:
        x = np.random.permutation(10000)
        num_images = 10000

    X_test = image_files[x[0:int(num_images*ratio)]]
    y_test = label_files[x[0:int(num_images*ratio)]]

    X_train = image_files[x[int(num_images*ratio):]]
    y_train = label_files[x[int(num_images*ratio):]]

    for x,y in zip(X_test,y_test):
        copyfile(os.path.join(image_path, x),
                 os.path.join(test_dir_img, x))
        
        copyfile(os.path.join(label_path, y),
                 os.path.join(test_dir_label, y))

    for x,y in zip(X_train,y_train):
        copyfile(os.path.join(image_path, x),
                 os.path.join(train_dir_img, x))
        
        copyfile(os.path.join(label_path, y),
                 os.path.join(train_dir_label, y))

if __name__=='__main__':
    main()
