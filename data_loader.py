import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets


# rgb_mean = {'cars': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044]}
# rgb_std = {'cars': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095]}


def train_data_loader(data_path, img_size, use_augment=False):
    if use_augment:
        data_transforms = transforms.Compose([
            transforms.RandomOrder([
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], .5),
                transforms.Compose([
                    transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], .5),
                    transforms.RandomApply([transforms.ColorJitter(hue=0.1)], .5),
                ])
            ]),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.125)], .5),
            transforms.RandomApply([transforms.RandomRotation(15)], .5),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_dataset = datasets.ImageFolder(data_path, data_transforms)

    return image_dataset


def test_data_loader(data_path):
    # return full path
    queries_path = [os.path.join(data_path, 'query', path) for path in os.listdir(os.path.join(data_path, 'query'))]
    references_path = [os.path.join(data_path, 'reference', path) for path in
                       os.listdir(os.path.join(data_path, 'reference'))]

    return queries_path, references_path


def test_data_generator(data_path, img_size):
    img_size = (img_size, img_size)
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_image_dataset = TestDataset(data_path, data_transforms)

    return test_image_dataset


class TestDataset(Dataset):
    def __init__(self, img_path_list, transform=None):
        self.img_path_list = img_path_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img_path, img

    def __len__(self):
        return len(self.img_path_list)


if __name__ == '__main__':
    query, refer = test_data_loader('./')
    print(query)
    print(refer)
