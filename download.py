# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.
import argparse
import multiprocessing.pool
import os
from io import BytesIO
from urllib.request import urlopen

import pandas as pd
from PIL import Image


class NoDeamonProcess(multiprocessing.Process):
    def _get_deamon(self):
        return False

    def _set_daemon(self):
        pass

    daemon = property(_get_deamon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    process = NoDeamonProcess


def download_image(key_urls):
    key, url = key_urls
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return 0

    try:
        response = urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image %s' % key)
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % filename)
        return 1

    print('Success: Image %s is download.' % filename)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Google Landmarks dataset')
    parser.add_argument('--data_file', type=str, help='path of data file')
    parser.add_argument('--output_dir', type=str, help='dir of downloaded images')

    opt = parser.parse_args()

    data_file, out_dir = opt.data_file, opt.output_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    csv_reader = pd.read_csv(data_file)
    kargs = [(k, u) for k, u in zip(csv_reader.id.tolist(), csv_reader.url.tolist())]
    pool = MyPool(4)
    pool.map(download_image, kargs)
