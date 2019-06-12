import json
import os

import numpy as np
from scipy.io import loadmat


# write json
def write_class_dic(mat_path, json_path):
    class_dic = {}
    cars_mat = loadmat(mat_path)
    info = cars_mat['class_names']
    with open(json_path, "w", encoding='utf-8') as f:
        for i in range(info.shape[1]):
            data = info[0, i]
            car_class = str(np.squeeze(data[0]))
            class_dic[str(i + 1)] = car_class
        f.write(json.dumps(class_dic))


# 写训练json
def write_train_json(mat_path, train_json_path, test_json_path, dic_path, img_root_path):
    train_dic = {}
    test_dic = {}
    train_mat = loadmat(mat_path)
    info = train_mat['annotations']
    dic = read_json(dic_path)  # 读取字典
    for i in range(info.shape[1]):
        data = info[0, i]
        test_sign = int(np.squeeze(data[-1]))
        car_label = int(np.squeeze(data[5]))
        car_classname = dic[str(car_label)]
        jpg_name = os.path.basename(str(np.squeeze(data[0])))
        img_path = os.path.join(img_root_path, jpg_name)
        if test_sign == 0:
            train_dic[jpg_name] = jpg_name
            train_dic[jpg_name] = {}
            train_dic[jpg_name]['classname'] = car_classname
            train_dic[jpg_name]['label'] = car_label
            train_dic[jpg_name]['path'] = img_path
        else:
            test_dic[jpg_name] = jpg_name
            test_dic[jpg_name] = {}
            test_dic[jpg_name]['classname'] = car_classname
            test_dic[jpg_name]['label'] = car_label
            test_dic[jpg_name]['path'] = img_path

    with open(train_json_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(train_dic))
    with open(test_json_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(test_dic))

    print('train preproccessing num: %s' % str(len(train_dic)))
    print('train preproccessing num: %s' % str(len(test_dic)))


# 写json文件
def write_json(path, dic):
    with open(path, "w", encoding='utf-8') as f:
        f.write(json.dumps(dic))


# 读取json文件
def read_json(path):
    with open(path, "r", encoding='utf-8') as f:
        dic = json.loads(f.read())
        # f.seek(0)
        # dic2 = json.load(f)
    return dic


if __name__ == '__main__':
    train_mat_path = 'data/cars196/cars_annos.mat'
    test_mat_path = 'data/cars196/cars_test_annos_withlabels.mat'
    cars_mat_path = 'data/cars196/cars_meta.mat'

    train_json_path = 'data/cars196/train.json'
    test_json_path = 'data/cars196/test.json'
    class_json_path = 'data/cars196/class_annotation.json'
    img_root_path = 'data/cars196/car_ims'

    # write_class_dic()

    write_train_json(train_mat_path, train_json_path, test_json_path, class_json_path, img_root_path)

    # write_test_json(test_mat_path, test_json_path, class_json_path)

    # with open(train_json_path, "r", encoding='utf-8') as f:
    #     json_file = json.loads(f.read())
    #     for i in json_file:
    #         print(i)
