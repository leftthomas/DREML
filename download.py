# Downloads images from the Google Landmarks dataset.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.
import argparse
import os
from io import BytesIO
from urllib.request import urlopen

import pandas as pd
from PIL import Image


def download_image(key, url):
    filename = '{}/{}.jpg'.format(out_dir, key)
    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = urlopen(url)
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

    opt = parser.parse_args()

    data_file, out_dir = 'data/{}.csv'.format(opt.data_type), 'data/{}'.format(opt.data_type)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    csv_reader = pd.read_csv(data_file)
    print('Download {} part of Google Landmarks dataset'.format(opt.data_type))
    for key, url in zip(csv_reader.id.tolist(), csv_reader.url.tolist()):
        download_image(key, url)
