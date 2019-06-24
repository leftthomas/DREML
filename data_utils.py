import json

import torch
from scipy.io import loadmat


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


def read_txt(path):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        data_1, data_2 = line.split()
        data[data_1] = data_2
    return data


def process_car_data(data_path):
    train_images, test_images = {}, {}
    annotations = loadmat('{}/cars_annos.mat'.format(data_path))['annotations'][0]
    for img in annotations:
        img_name, img_label = str(img[0][0]), str(img[-2][0][0])
        if int(img_label) < 99:
            if img_label in train_images:
                train_images[img_label].append('{}/{}'.format(data_path, img_name))
            else:
                train_images[img_label] = ['{}/{}'.format(data_path, img_name)]
        else:
            if img_label in test_images:
                test_images[img_label].append('{}/{}'.format(data_path, img_name))
            else:
                test_images[img_label] = ['{}/{}'.format(data_path, img_name)]
    torch.save({'tra': train_images, 'test': test_images}, '{}/{}'.format(data_path, train_image_json))


def process_cub_data(data_path):
    images = read_txt('{}/images.txt'.format(data_path))
    labels = read_txt('{}/image_class_labels.txt'.format(data_path))
    train_images, test_images = {}, {}
    for img_id, img_name in images.items():
        if int(labels[img_id]) < 101:
            train_images['{}/images/{}'.format(data_path, img_name)] = labels[img_id]
        else:
            test_images['{}/images/{}'.format(data_path, img_name)] = labels[img_id]
    write_json(train_images, '{}/{}'.format(data_path, train_image_json))
    write_json(test_images, '{}/{}'.format(data_path, test_image_json))


def process_sop_data(data_path):
    train_images, test_images = {}, {}
    for index, line in enumerate(open('{}/Ebay_train.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, label, _, img_name = line.split()
            train_images['{}/{}'.format(data_path, img_name)] = label
    write_json(train_images, '{}/{}'.format(data_path, train_image_json))

    for index, line in enumerate(open('{}/Ebay_test.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, label, _, img_name = line.split()
            test_images['{}/{}'.format(data_path, img_name)] = label
    write_json(test_images, '{}/{}'.format(data_path, test_image_json))


if __name__ == '__main__':
    train_image_json, test_image_json = 'data_dict_emb.pth', 'test_images.json'
    process_car_data('data/cars')
    # process_cub_data('data/cub')
    # process_sop_data('data/sop')
