# DCN
A PyTorch implementation of Diverse Capsule Network based on the paper [Diverse Capsule Network for Image Retrieval]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
- capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Datasets
[cars196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [cub200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
and [sop](http://cvgl.stanford.edu/projects/lifted_struct/) are used in this repo.

You should download these datasets by yourself, and extract them into `data` directory, make sure the dir names are 
`car`, `cub` and `sop`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 12
optional arguments:
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop'])
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 32]
--num_epochs                  train epochs number [default value is 12]
```
Visdom now can be accessed by going to `127.0.0.1:8097/$data_type` in your browser.

## Results
The train loss, recall and test loss, recall are showed on visdom.

### car
![result](results/car.png)

### cub
![result](results/cub.png)

### sop
![result](results/sop.png)

