# Downloads images from the Google Landmarks dataset.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.
import argparse
import os
from io import BytesIO
from urllib.request import Request
from urllib.request import urlopen

import pandas as pd
from PIL import Image
from joblib import Parallel
from joblib import delayed


def download_image(key, url):
    filename = '{}/{}.jpg'.format(out_dir, key)
    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        request = Request(url)
        response = urlopen(request)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}.'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}.'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB.'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}.'.format(filename))
        return 1

    print('Success: Image {} is download.'.format(filename))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Google Landmarks dataset')
    parser.add_argument('--data_type', default='index', type=str, choices=['index', 'train', 'test'],
                        help='type of data')
    parser.add_argument('--n_jobs', default=24, type=int, help='number of parallel jobs')

    opt = parser.parse_args()

    data_file, out_dir = 'data/{}.csv'.format(opt.data_type), 'data/{}'.format(opt.data_type)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    csv_reader = pd.read_csv(data_file)
    print('Download {} part of Google Landmarks dataset'.format(opt.data_type))
    Parallel(n_jobs=opt.n_jobs)(delayed(download_image)(row['id'], row['url']) for i, row in csv_reader.iterrows())
    # clean the corrupted images
    print('Check {} part of Google Landmarks dataset, if the image is corrupted, it will be deleted'.format(
        opt.data_type))
    for image_name in sorted(os.listdir(out_dir)):
        try:
            im = Image.open(image_name)
            print('{} is success saved.'.format(image_name))
        except IOError:
            # damaged
            os.remove(image_name)
            print('{} is corrupted and removed.'.format(image_name))
