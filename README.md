# DCN
A PyTorch implementation of Diverse Capsule Network based on the paper [Diverse Capsule Network for Image Retrieval]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
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
python train.py --num_epochs 12
optional arguments:
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop'])
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 128]
--num_epochs                  train epochs number [default value is 12]
--ensemble_size               ensemble model size [default value is 12]
--meta_class_size             meta class size [default value is 12]
```

## Benchmarks
Adam optimizer is used with learning rate scheduling. The models are trained with 12 epochs and batch size of 128 on one 
NVIDIA Tesla V100 (32G) GPU.

The images are preprocessed with random resize, random crop, random horizontal flip, and normalize.
Here is the recall details:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Cars196(CNN)</th>
      <th>CUB200(CNN)</th>
      <th>SOP(CNN)</th>
      <th>Cars196(Capsule)</th>
      <th>CUB200(Capsule)</th>
      <th>SOP(Capsule)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

