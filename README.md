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
TODO

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'cars'](choices=['cars', 'cub', 'sop'])
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 32]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/$data_type` in your browser.

## Results
The train loss, accuracy, recall and test loss, accuracy, recall are showed on visdom.

### cars
![result](results/cars.png)

### cub
![result](results/cub.png)

### sop
![result](results/sop.png)

