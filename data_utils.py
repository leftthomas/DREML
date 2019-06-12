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
    classes, trains, tests = {}, {}, {}
    annos_mat = loadmat('{}/cars_annos.mat'.format(data_path))
    class_names, annotations = annos_mat['class_names'][0], annos_mat['annotations'][0]
    for index, class_name in enumerate(class_names):
        classes[index + 1] = str(class_name[0])
    write_json(classes, '{}/{}'.format(data_path, class_json))

    for img in annotations:
        img_name, img_label, flag = str(img[0][0]), str(img[-2][0][0]), int(img[-1][0][0])
        if flag == 0:
            trains['{}/{}'.format(data_path, img_name)] = img_label
        else:
            tests['{}/{}'.format(data_path, img_name)] = img_label
    write_json(trains, '{}/{}'.format(data_path, train_json))
    write_json(tests, '{}/{}'.format(data_path, test_json))


def process_cub_data(data_path):
    classes = read_txt('{}/classes.txt'.format(data_path))
    write_json(classes, '{}/{}'.format(data_path, class_json))
    images = read_txt('{}/images.txt'.format(data_path))
    splits = read_txt('{}/train_test_split.txt'.format(data_path))
    labels = read_txt('{}/image_class_labels.txt'.format(data_path))
    trains, tests = {}, {}
    for index, img_name in enumerate(images.values()):
        if splits[str(index + 1)] == '1':
            trains['{}/{}'.format(data_path, img_name)] = labels[str(index + 1)]
        else:
            tests['{}/{}'.format(data_path, img_name)] = labels[str(index + 1)]
    write_json(trains, '{}/{}'.format(data_path, train_json))
    write_json(tests, '{}/{}'.format(data_path, test_json))


def process_sop_data(data_path):
    annos_mat = '{}/cars_annos.mat'.format(data_path)
    meta_mat = '{}/cars_meta.mat'.format(data_path)
    imgs_path = '{}/car_ims'.format(data_path)
    write_class(meta_mat, '{}/{}'.format(data_path, class_json))


if __name__ == '__main__':
    train_json, test_json, class_json = 'train.json', 'test.json', 'class.json'
    process_car_data('data/cars')
    process_cub_data('data/cub')
    process_sop_data('data/sop')

    # write_class_dic()

    write_train(train_mat_path, train_json_path, test_json_path, class_json_path, img_root_path)

    # write_test_json(test_mat_path, test_json_path, class_json_path)

    # with open(train_json_path, "r", encoding='utf-8') as f:
    #     json_file = json.loads(f.read())
    #     for i in json_file:
    #         print(i)
