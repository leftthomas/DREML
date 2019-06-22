import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from numpy import linalg as LA
from torchvision.models import resnet50

import utils


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # backbone
        basic_model, layers = resnet50(pretrained=True).eval(), []
        for name, module in basic_model.named_children():
            if isinstance(module, nn.Linear) or isinstance(module, nn.AdaptiveAvgPool2d):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=7)
        x = x.view(x.size(0), -1)
        return x


def extractor(data_path, feat_extractor):
    image_names, features = os.listdir(data_path), []
    # extract features
    for img_name in image_names:
        img_path = os.path.join(data_path, img_name)
        img = utils.get_transform('cars', 'test')(Image.open(img_path).convert('RGB'))
        img = img.unsqueeze(0).cuda()
        feat = feat_extractor(img)
        feat = feat.detach().cpu().numpy()
        feat = feat / LA.norm(feat)
        features.append(feat)
    features = np.concatenate(features, axis=0)
    return features, image_names


if __name__ == '__main__':
    feature_extractor = FeatureExtractor().cuda()
    # extract features from reference dataset
    reference_feats, reference_image_list = extractor('data/reference', feature_extractor)
    query_feats, query_image_list = extractor('data/test', feature_extractor)

    scores = np.dot(query_feats, reference_feats.T)
    sort_ind = np.argsort(scores)[0][::-1]
    scores = scores[0, sort_ind]

    max_res = 10
    res_im_list = [reference_image_list[index] for index in sort_ind[0:max_res]]
    print('top {} images in order are: {}'.format(max_res, res_im_list))
