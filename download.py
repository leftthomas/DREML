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


def download_image(index, key, url):
    filename = '{}/{}.jpg'.format(out_dir, key)
    if os.path.exists(filename):
        print('Index: %-20s ID: %-20s Status: %-20s' % (index, key, 'Already Exists.'))
        return 0

    try:
        headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}
        request = Request(url, headers=headers)
        response = urlopen(request)
        image_data = response.read()
    except Exception as e:
        print('Index: %-20s ID: %-20s Status: %-20s Reason: %-20s' % (index, key, 'Request Error', '{}.'.format(e)))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except Exception as e:
        print('Index: %-20s ID: %-20s Status: %-20s Reason: %-20s' % (index, key, 'Open Error', '{}.'.format(e)))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except Exception as e:
        print('Index: %-20s ID: %-20s Status: %-20s Reason: %-20s' % (index, key, 'Convert Error', '{}.'.format(e)))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except Exception as e:
        print('Index: %-20s ID: %-20s Status: %-20s Reason: %-20s' % (index, key, 'Save Error', '{}.'.format(e)))
        return 1

    print('Index: %-20s ID: %-20s Status: %-20s' % (index, key, 'Success Saved.'))
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
    num_data = len(csv_reader)
    print('Download {} part of Google Landmarks dataset'.format(opt.data_type))
    Parallel(n_jobs=opt.n_jobs)(
        delayed(download_image)('{}/{}'.format(str(i + 1), str(num_data)), row['id'], row['url']) for i, row in
        csv_reader.iterrows())
    # clean the corrupted images
    print('Check {} part of Google Landmarks dataset, if the image is corrupted, it will be deleted'.format(
        opt.data_type))
    for image_name in sorted(os.listdir(out_dir)):
        try:
            im = Image.open('{}/{}'.format(out_dir, image_name))
            print('Image-Name: %-20s Status: %-20s' % (image_name, 'Success Saved.'))
        except IOError:
            # damaged
            os.remove('{}/{}'.format(out_dir, image_name))
            print('Image-Name: %-20s Status: %-20s' % (image_name, 'Have Corrupted.'))
