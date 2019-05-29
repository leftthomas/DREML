# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.
import argparse
import multiprocessing
import os
from io import StringIO
from urllib.request import urlopen

import pandas as pd
from PIL import Image


def parse_data(data_file_name):
    csv_reader = pd.read_csv(data_file_name)
    keys, urls = csv_reader['id'].to_list(), csv_reader['url'].to_list()
    return keys, urls


def download_image(key_urls):
    key, url = key_urls[0], key_urls[1]
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return

    try:
        response = urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(StringIO(image_data))
    except:
        print('Warning: Failed to parse image %s' % key)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % filename)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Google Landmarks dataset')
    parser.add_argument('--data_file', type=str, help='path of data file')
    parser.add_argument('--output_dir', type=str, help='dir of downloaded images')

    opt = parser.parse_args()

    data_file, out_dir = opt.data_file, opt.output_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    keys, urls = parse_data(data_file)
    pool = multiprocessing.Pool(processes=50)
    pool.map(download_image, [keys, urls])
