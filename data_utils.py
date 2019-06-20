import json

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
    train_images, test_images, class_names = {}, {}, {}
    annos_mat = loadmat('{}/cars_annos.mat'.format(data_path))
    classes, annotations = annos_mat['class_names'][0], annos_mat['annotations'][0]
    for index, class_name in enumerate(classes):
        class_names[index + 1] = str(class_name[0])
    write_json(class_names, '{}/{}'.format(data_path, class_name_json))

    for img in annotations:
        img_name, img_label = str(img[0][0]), str(img[-2][0][0])
        if int(img_label) < 99:
            train_images['{}/{}'.format(data_path, img_name)] = img_label
        else:
            test_images['{}/{}'.format(data_path, img_name)] = img_label
    write_json(train_images, '{}/{}'.format(data_path, train_image_json))
    write_json(test_images, '{}/{}'.format(data_path, test_image_json))


def process_cub_data(data_path):
    classes = read_txt('{}/classes.txt'.format(data_path))
    write_json(classes, '{}/{}'.format(data_path, class_name_json))
    images = read_txt('{}/images.txt'.format(data_path))
    splits = read_txt('{}/train_test_split.txt'.format(data_path))
    labels = read_txt('{}/image_class_labels.txt'.format(data_path))
    trains, tests = {}, {}
    for index, img_name in enumerate(images.values()):
        if splits[str(index + 1)] == '1':
            trains['{}/{}'.format(data_path, img_name)] = labels[str(index + 1)]
        else:
            tests['{}/{}'.format(data_path, img_name)] = labels[str(index + 1)]
    write_json(trains, '{}/{}'.format(data_path, train_image_json))
    write_json(tests, '{}/{}'.format(data_path, test_image_json))


def process_sop_data(data_path):
    class_names, classes, trains, tests = set(), {}, {}, {}
    for index, line in enumerate(open('{}/Ebay_train.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, _, label, img_name = line.split()
            trains['{}/{}'.format(data_path, img_name)] = label
            class_names.add(img_name.split('/')[0].replace('_final', ''))
    for index, line in enumerate(open('{}/Ebay_test.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, _, label, img_name = line.split()
            tests['{}/{}'.format(data_path, img_name)] = label
            class_names.add(img_name.split('/')[0].replace('_final', ''))

    for index, class_name in enumerate(sorted(class_names)):
        classes[index + 1] = class_name
    write_json(classes, '{}/{}'.format(data_path, class_name_json))
    write_json(trains, '{}/{}'.format(data_path, train_image_json))
    write_json(tests, '{}/{}'.format(data_path, test_image_json))


if __name__ == '__main__':
    train_image_json, test_image_json, class_name_json = 'train_images.json', 'test_images.json', 'class_names.json'
    process_car_data('data/cars')
    process_cub_data('data/cub')
    process_sop_data('data/sop')

