from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

# path to images dataset. Recommended to write down entire path to avoid errors (i.e "C:/User/Desktop.../raw/)
data_path = 'C:\\Users\\Reem\\Projects\\knee-MRI-segmentation\\data\\raw'
data_dest = os.path.join(os.getcwd,'data')

image_rows = 384
image_cols = 384


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tiff'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    img_train_dest = os.path.join(data_dest, 'imgs_train.npy')
    img_train_mask_dest = os.path.join(data_dest, 'imgs_mask_train.npy')

    np.save(img_train_dest, imgs)
    np.save(img_train_mask_dest, imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    img_train_dest = os.path.join(data_dest, 'imgs_train.npy')
    img_train_mask_dest = os.path.join(data_dest, 'imgs_mask_train.npy')
    imgs_train = np.load(img_train_dest)
    imgs_mask_train = np.load(img_train_mask_dest)
    return imgs_train, imgs_mask_train


def create_validate_data():
    validate_data_path = os.path.join(data_path, 'validate')
    images = os.listdir(validate_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating validation images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tiff'
        img_val_id = int(image_name.split('.')[0])
        img = imread(os.path.join(validate_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(validate_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    img_val_dest = os.path.join(data_dest, 'imgs_validate.npy')
    img_val_mask_dest = os.path.join(data_dest, 'imgs_mask_validate.npy')

    np.save(img_val_dest, imgs)
    np.save(img_val_mask_dest, imgs_mask)
    print('Saving to .npy files done.')


def load_validate_data():
    img_val_dest = os.path.join(data_dest, 'imgs_validate.npy')
    img_val_mask_dest = os.path.join(data_dest, 'imgs_mask_validate.npy')
    imgs_validate = np.load(img_val_dest)
    imgs_mask_validate = np.load(img_val_mask_dest)
    return imgs_validate, imgs_mask_validate


def create_test_data():
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int64)

    i = 0
    print('-'*30)
    print('Creating testing images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tiff'
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(test_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask
        imgs_id[i] = img_id

        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    img_test_dest = os.path.join(data_dest, 'imgs_test.npy')
    img_test_mask_dest = os.path.join(data_dest, 'imgs_mask_test.npy')
    img_test_id_dest = os.path.join(data_dest, 'imgs_id_test.npy')
    np.save(img_test_dest, imgs)
    np.save(img_test_mask_dest, imgs_mask)
    np.save(img_test_id_dest, imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    img_test_dest = os.path.join(data_dest, 'imgs_test.npy')
    img_test_mask_dest = os.path.join(data_dest, 'imgs_mask_test.npy')
    img_test_id_dest = os.path.join(data_dest, 'imgs_id_test.npy')
    imgs_test = np.load(img_test_dest)
    imgs_mask_test = np.load(img_test_mask_dest)
    imgs_id = np.load(img_test_id_dest)
    return imgs_test, imgs_id, imgs_mask_test

if __name__ == '__main__':
    create_train_data()
    create_validate_data()
    create_test_data()


# In[3]:
